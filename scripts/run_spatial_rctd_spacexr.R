suppressPackageStartupMessages({
  library(Matrix)
  library(spacexr)
  library(SpatialExperiment)
  library(SummarizedExperiment)
  library(S4Vectors)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0 || idx[1] == length(args)) {
    return(default)
  }
  args[idx[1] + 1]
}

has_flag <- function(flag) {
  any(args == flag)
}

input_dir <- normalizePath(get_arg("--input-dir", "out/emtab13530_spatial_bach1_nod/rctd_spacexr_inputs"), mustWork = TRUE)
output_dir <- get_arg("--output-dir", "out/emtab13530_spatial_bach1_nod/rctd_spacexr_results")
mode <- get_arg("--mode", "multi")
max_cores <- as.integer(get_arg("--max-cores", "4"))
max_multi_types <- as.integer(get_arg("--max-multi-types", "4"))
tissue_groups_arg <- get_arg("--tissue-groups", "")
allowed_tissue_groups <- trimws(unlist(strsplit(tissue_groups_arg, ",")))
allowed_tissue_groups <- allowed_tissue_groups[nzchar(allowed_tissue_groups)]
resume <- has_flag("--resume")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "weights"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "coldata"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "rds"), recursive = TRUE, showWarnings = FALSE)

read_lines <- function(path) {
  readLines(path, warn = FALSE)
}

read_counts <- function(path, genes_path, columns_path) {
  counts <- readMM(gzfile(path))
  genes <- read_lines(genes_path)
  columns <- read_lines(columns_path)
  rownames(counts) <- make.unique(genes)
  colnames(counts) <- make.unique(columns)
  as(counts, "dgCMatrix")
}

sanitize_df_for_csv <- function(df) {
  df[] <- lapply(df, function(col) {
    if (is.list(col)) {
      vapply(col, function(value) {
        paste(as.character(unlist(value)), collapse = ";")
      }, character(1))
    } else {
      col
    }
  })
  df
}

reference_dir <- file.path(input_dir, "reference")
ref_counts <- read_counts(
  file.path(reference_dir, "reference_counts.mtx.gz"),
  file.path(reference_dir, "genes.tsv"),
  file.path(reference_dir, "cells.tsv")
)
cell_types <- read.csv(file.path(reference_dir, "cell_types.csv"), stringsAsFactors = FALSE)
cell_types <- cell_types[match(colnames(ref_counts), cell_types$cell), , drop = FALSE]
reference_se <- SummarizedExperiment(
  assays = list(counts = ref_counts),
  colData = DataFrame(cell_type = factor(cell_types$cell_type), row.names = colnames(ref_counts))
)

manifest <- read.csv(file.path(input_dir, "spatial_sample_manifest.csv"), stringsAsFactors = FALSE)
manifest <- manifest[manifest$status == "ready", , drop = FALSE]
if (length(allowed_tissue_groups) > 0 && "tissue_group" %in% colnames(manifest)) {
  manifest <- manifest[manifest$tissue_group %in% allowed_tissue_groups, , drop = FALSE]
}

run_rows <- list()
for (i in seq_len(nrow(manifest))) {
  sample <- manifest$sample[i]
  sample_dir <- manifest$path[i]
  out_weights <- file.path(output_dir, "weights", paste0(sample, "_rctd_weights.csv.gz"))
  out_coldata <- file.path(output_dir, "coldata", paste0(sample, "_rctd_coldata.csv.gz"))
  out_rds <- file.path(output_dir, "rds", paste0(sample, "_rctd_result.rds"))

  if (resume && file.exists(out_weights) && file.info(out_weights)$size > 0) {
    run_rows[[length(run_rows) + 1]] <- data.frame(
      sample = sample,
      status = "skipped_existing",
      message = "",
      stringsAsFactors = FALSE
    )
    next
  }

  message("Running RCTD for ", sample)
  result_row <- tryCatch({
    sp_counts <- read_counts(
      file.path(sample_dir, "spatial_counts.mtx.gz"),
      file.path(sample_dir, "genes.tsv"),
      file.path(sample_dir, "spots.tsv")
    )
    meta <- read.csv(file.path(sample_dir, "spot_metadata.csv"), stringsAsFactors = FALSE)
    meta <- meta[match(colnames(sp_counts), meta$spot), , drop = FALSE]
    coords <- as.matrix(meta[, c("x", "y")])
    rownames(coords) <- colnames(sp_counts)
    spatial_spe <- SpatialExperiment(
      assays = list(counts = sp_counts),
      colData = DataFrame(meta, row.names = colnames(sp_counts)),
      spatialCoords = coords
    )
    rctd_data <- createRctd(
      spatial_spe,
      reference_se,
      cell_type_col = "cell_type",
      require_int = TRUE,
      gene_cutoff = 0.00025,
      fc_cutoff = 0.5,
      gene_cutoff_reg = 0.00025,
      fc_cutoff_reg = 0.75,
      gene_obs_min = 3,
      pixel_count_min = 10,
      UMI_min = 100,
      ref_UMI_min = 100,
      ref_n_cells_min = 25,
      ref_n_cells_max = 10000
    )
    rctd_res <- runRctd(
      rctd_data,
      rctd_mode = mode,
      max_cores = max_cores,
      max_multi_types = max_multi_types
    )
    assay_name <- if ("weights" %in% assayNames(rctd_res)) "weights" else assayNames(rctd_res)[1]
    weights <- as.matrix(assay(rctd_res, assay_name))
    weights_out <- as.data.frame(t(weights))
    weights_out$spot <- rownames(weights_out)
    weights_out <- weights_out[, c("spot", setdiff(colnames(weights_out), "spot")), drop = FALSE]
    write.csv(weights_out, gzfile(out_weights), row.names = FALSE)

    cd <- sanitize_df_for_csv(as.data.frame(colData(rctd_res)))
    cd$spot <- rownames(cd)
    cd <- cd[, c("spot", setdiff(colnames(cd), "spot")), drop = FALSE]
    write.csv(cd, gzfile(out_coldata), row.names = FALSE)
    saveRDS(rctd_res, out_rds)
    data.frame(
      sample = sample,
      status = "completed",
      message = "",
      n_spots_input = ncol(sp_counts),
      n_genes_input = nrow(sp_counts),
      n_spots_output = ncol(weights),
      n_cell_types = nrow(weights),
      assay_name = assay_name,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      sample = sample,
      status = "failed",
      message = conditionMessage(e),
      stringsAsFactors = FALSE
    )
  })
  run_rows[[length(run_rows) + 1]] <- result_row
  write.csv(do.call(rbind, run_rows), file.path(output_dir, "rctd_run_status.csv"), row.names = FALSE)
}

status <- do.call(rbind, run_rows)
write.csv(status, file.path(output_dir, "rctd_run_status.csv"), row.names = FALSE)
message("RCTD run status written to ", file.path(output_dir, "rctd_run_status.csv"))
