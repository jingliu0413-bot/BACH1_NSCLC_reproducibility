# Codex代码上传交班文件

## 任务目标

在另一台设备上，将本目录作为独立代码仓库上传到GitHub或同类公共仓库；稿件接收后再用Zenodo生成永久DOI。不要上传整个原项目目录。

## 交付位置

- 发布目录：`I:\1-NSCLC\BACH1_NSCLC_reproducibility`
- 压缩包：`I:\1-NSCLC\BACH1_NSCLC_reproducibility.zip`
- 原始项目约51 GB；发布目录仅包含代码、文档、环境和轻量源数据。

## 已完成的整理

1. 从`out`目录抽取了最终主流程使用的分析和作图脚本。
2. 将写死的`I:\1-NSCLC`路径替换为`NSCLC_PROJECT_ROOT`等环境变量。
3. 补齐GSE274934九个GEX矩阵合并、三类CNV结果合并、pySCENIC精确命令、公开数据下载和外部数据库下载脚本。
4. 将pySCENIC/scATAC主交集限定为TSS +/-10 kb，不在发布主流程中展示100 kb交集富集。
5. 附带Figure 1-5、Figure S1和NOD-like receptor locus图的轻量源数据。
6. Figure S1发布脚本已按最终要求删除原S1C和S1E，仅保留流程、CNV burden UMAP和cluster阈值图，顺序编号为a-c。
7. 所有Python脚本已通过`compileall`语法检查，并扫描确认没有个人绝对路径。

## 必须注意的现状

- 原项目`out\nature_story_4figures_10kb\figures\发表图\Figure S1.pdf`及现有双语图例仍是旧的a-e版本。此次没有覆盖用户已经修改的稿件或发表图。
- 如果准备正式投稿，应使用发布目录中的`plot_cnv_supplement_figure_s1.py`重新生成三面板Figure S1，并只同步修改Figure S1图例。未经用户明确指示，不要修改稿件其他内容。
- `CODE_AVAILABILITY.md`中的GitHub URL和Zenodo DOI仍是占位符，上传后必须替换。
- `LICENSE`暂以“BACH1 NSCLC study authors”为权利人。公开前应由用户确认是否保留MIT许可证及权利人写法。
- 本次新增的下载、GSE274934合并、CNV合并和pySCENIC命令封装脚本完成了静态检查，但没有从零进行全量51 GB重跑。原始分析脚本和原有结果已经实际运行过。
- Enrichr的`*_2026`在线库可能更新，因此仓库保留了本次报告图的source data；重跑结果应核对库版本和日期。

## 科学口径不可改动

- pySCENIC：233个motif-pruned BACH1 regulon候选基因。
- scATAC TSS +/-10 kb：3,830个高置信motif相关候选基因。
- 两者交集：65个“双证据候选基因”，不能写成已证实直接靶基因。
- motif不等于BACH1占位，可及性不等于调控，空间共定位不等于因果。
- 空间结果重点是肿瘤中BACH1阳性区域扩展和BACH1/NOD共定位增强；NOD-like pathway score单独比较不显著。
- BACH1 pySCENIC是单转录因子定向分析，不是全TF无偏网络重建。

## 另一台Codex的操作顺序

1. 解压`BACH1_NSCLC_reproducibility.zip`，只操作解压后的发布目录。
2. 阅读`README.md`、`docs/WORKFLOW.md`和本文件。
3. 运行静态检查：

```powershell
python -m compileall -q scripts
rg -n "I:/1-NSCLC|I:\\1-NSCLC|C:\\Users" .
Get-ChildItem -Recurse -File | Sort-Object Length -Descending | Select-Object -First 20 FullName,Length
```

4. 确认没有`.h5ad`、`.h5`、`.loom`、原始数据、完整数据库、虚拟环境、稿件、审稿意见或个人路径。
5. 向用户确认GitHub账号、仓库名称、公开/私有状态和许可证权利人；不要自行猜测。
6. 初始化并提交：

```powershell
git init
git add .
git commit -m "Release reproducible analysis code"
```

7. 用户确认远程仓库后再创建和推送。若使用GitHub CLI，可执行：

```powershell
gh repo create <repository-name> --private --source . --remote origin --push
```

8. 投稿审稿阶段可先使用私有仓库审稿链接；接受前转为公开，并通过Zenodo归档release生成DOI。
9. 将最终URL和DOI写回`CODE_AVAILABILITY.md`以及稿件Code availability段落。

## 上传范围

应上传：`scripts`、`environment`、`docs`、`resources`中的说明和校验文件、`source_data`、`README.md`、`CODE_AVAILABILITY.md`、`LICENSE`、`.gitignore`和校验清单。

不要上传：原项目`data`、`out`、`envs`、`tmp_pyscenic_pkg`、完整`resources`数据库、发表图文件夹、稿件DOCX/PDF、`BACH1-recommendations and comments 2026 08 08.docx`或任何Codex会话记录。
