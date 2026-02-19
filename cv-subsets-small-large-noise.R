library(data.table)
n.folds <- 6
max.N <- 2000
N <- max.N*n.folds/(n.folds-1)
abs.x <- 3*pi
set.seed(2)
norm01 <- function(z,ref=z)(z-min(ref))/(max(ref)-min(ref))
(grid.dt <- data.table(
  raw=seq(-abs.x,abs.x, l=201),
  y=0 #for mlr3
)[, x := norm01(raw)][])
x.vec <- runif(N, -abs.x, abs.x)
str(x.vec)
reg.pattern.list <- list(
  sin=sin)
standard.deviation.vec <- c(
  easy=0.1,
  hard=1.7)
reg.task.list <- list()
reg.data.list <- list()
grid.signal.dt.list <- list()
for(signal in names(reg.pattern.list)){
  f <- reg.pattern.list[[signal]]
  for(difficulty in names(standard.deviation.vec)){
    standard.deviation <- standard.deviation.vec[[difficulty]]
    task_id <- paste(signal, difficulty)
    signal.vec <- f(x.vec)
    y <- signal.vec+rnorm(N,sd=standard.deviation)
    task.dt <- data.table(
      x=norm01(x.vec,grid.dt$raw), y)
    reg.data.list[[paste(difficulty, task_id)]] <- data.table(
      difficulty,
      signal,
      task_id,
      task.dt)
    reg.task.list[[paste(difficulty, task_id)]] <- mlr3::TaskRegr$new(
      task_id, task.dt, target="y"
    )
    grid.signal.dt.list[[paste(difficulty, task_id)]] <- data.table(
      difficulty,
      signal,
      task_id,
      algorithm="ideal",
      x=grid.dt$x,
      y=f(grid.dt$raw)
    )
  }
}
(reg.data <- rbindlist(reg.data.list))
(grid.signal.dt <- rbindlist(grid.signal.dt.list))
algo.colors <- c(
  featureless="blue",
  rpart="red",
  ideal="black")
if(require(animint2)){
  ggplot()+
    theme_bw()+
    geom_point(aes(
      x, y),
      fill="white",
      color="grey",
      data=reg.data)+
    geom_line(aes(
      x, y, color=algorithm),
      size=2,
      data=grid.signal.dt)+
    scale_color_manual(values=algo.colors)+
    facet_grid(. ~ difficulty, labeller=label_both)
}

reg_size_cv <- mlr3resampling::ResamplingVariableSizeTrainCV$new()
n.train.sizes <- 18
reg_size_cv$param_set$values$train_sizes <- n.train.sizes
reg_size_cv$param_set$values$folds <- n.folds
reg_size_cv$param_set$values$random_seeds <- 1
reg_size_cv$instantiate(reg.task.list[[1]])#required for consistent folds across tasks.

(reg.learner.list <- list(
  if(requireNamespace("rpart"))mlr3::LearnerRegrRpart$new(),
  mlr3::LearnerRegrFeatureless$new()))
(reg.bench.grid <- mlr3::benchmark_grid(
  reg.task.list,
  reg.learner.list,
  reg_size_cv))

if(FALSE){
  if(require(future))plan("multisession")
}
if(require(lgr))get_logger("mlr3")$set_threshold("warn")
(reg.bench.result <- mlr3::benchmark(
  reg.bench.grid, store_models = TRUE))

reg.bench.score <- nc::capture_first_df(
  mlr3resampling::score(reg.bench.result),
  task_id=list(
    signal=".*?",
    " ",
    difficulty=".*"))[
, signal_difficulty_Ntrain := paste(signal,difficulty,train_size),
][]
train_size_vec <- unique(reg.bench.score$train_size)

grid.task <- mlr3::TaskRegr$new("grid", grid.dt, target="y")
pred.dt.list <- list()
point.dt.list <- list()
for(score.i in 1:nrow(reg.bench.score)){
  reg.bench.row <- reg.bench.score[score.i]
  task.dt <- data.table(
    reg.bench.row$task[[1]]$data(),
    reg.bench.row$resampling[[1]]$instance$id.dt)
  set.ids <- data.table(
    Set=c("test","train")
  )[
  , data.table(row_id=reg.bench.row[[Set]][[1]])
  , by=Set]
  i.points <- set.ids[
    task.dt, on="row_id"
  ][
    is.na(Set), Set := "unused"
  ]
  point.id <- reg.bench.row[, paste(signal_difficulty_Ntrain, test.fold, algorithm)]
  point.dt.list[[point.id]] <- data.table(
    reg.bench.row[, .(signal_difficulty_Ntrain, test.fold, algorithm)],
    i.points)
  i.learner <- reg.bench.row$learner[[1]]
  pred.dt.list[[score.i]] <- data.table(
    reg.bench.row[, .(
      signal_difficulty_Ntrain, signal, difficulty, train_size, test.fold, algorithm
    )],
    as.data.table(
      i.learner$predict(grid.task)
    )[, .(x=grid.dt$x, y=response)]
  )
}
(pred.dt <- rbindlist(pred.dt.list))
(point.dt <- rbindlist(point.dt.list)[algorithm=="featureless"])
(upred <- unique(pred.dt[, .(signal_difficulty_Ntrain, signal, difficulty, train_size)]))
signal.dt <- upred[
  grid.signal.dt, on=.(signal,difficulty), allow.cartesian=TRUE]

point.dt[grepl(" 10$", signal_difficulty_Ntrain) & test.fold==1 & Set=="train"][, .SD[1:2], by=.(signal_difficulty_Ntrain)]

algo.sizes <- c(
  ideal=1,
  featureless=4,
  rpart=2)
(reg.bench.wide <- dcast(
  reg.bench.score,
  signal + difficulty + train_size + algorithm + signal_difficulty_Ntrain ~ .,
  list(mean, sd, length, min, max),
  value.var=c("regr.mse")))
reg.bench.test <- dcast(
  reg.bench.score[, log10.mse := log10(regr.mse)],
  signal + difficulty + train_size + test.fold + signal_difficulty_Ntrain ~ algorithm,
  value.var=c("log10.mse"))
rect.x <- seq(1,log10(max.N),l=n.train.sizes)
seq.diff <- diff(rect.x)[1]/2
test.proposed <- reg.bench.test[, {
  paired <- t.test(rpart, featureless, alternative="two.sided", paired=TRUE)
  unpaired <- t.test(rpart, featureless, alternative="two.sided", paired=FALSE)
  data.table(
    mean.of.diff=paired$estimate, p.paired=paired$p.value, p.value=unpaired$p.value,
    mean.rpart=unpaired$estimate[1], mean.featureless=unpaired$estimate[2], p.unpaired=unpaired$p.value)
}, keyby=.(signal,difficulty,train_size,signal_difficulty_Ntrain)
][, `:=`(
  difference=ifelse(
    is.nan(p.value) | p.value>0.05, "not significant", "significant"),
  xmin=10^(rect.x-seq.diff),
  xmax=10^(rect.x+seq.diff)
), by=.(signal,difficulty)][]
reg.bench.join <- reg.bench.wide[
  test.proposed[, .(signal_difficulty_Ntrain,signal,difficulty,train_size,difference)],
  on=.NATURAL]
mid.x <- 10^((max(rect.x)+min(rect.x))/2)
data.color <- "grey50"
mse.limits <- c(0.04, 5)
mse.breaks <- c(0.05, 0.2, 0.8, 3.2)
Toff <- 1.2
Tbrk <- c(0,0.5,1)
Tbreaks <- c(Tbrk,Tbrk+Toff)
Tlabels <- c(Tbrk,Tbrk)
unused.x <- 1.1
unused.y.point <- -3
unused.y.text <- -4
Set.y.text <- 4
Tpred <- function(DT){
  if(! "Set" %in% names(DT)){
    DT <- data.table(Set=c("train","test"))[, data.table(DT), by=Set]
  }
  data.table(DT)[
  , x := ifelse(Set=="test",0,Toff)+x
  ][
    Set=="unused", `:=`(x=unused.x, y=unused.y.point)
  ][]
}
(data.sizes <- point.dt[, .(N=.N), by=.(signal_difficulty_Ntrain, test.fold, Set)])


viz <- animint(
  title="Samples required to learn non-trivial regression model",
  video="https://vimeo.com/1051473773",
  overview=ggplot()+
    ggtitle("Select signal, difficulty, Ntrain")+
    theme_bw()+
    theme_animint(width=600, height=300)+
    geom_ribbon(aes(
      train_size,
      ymin=regr.mse_mean-regr.mse_sd,
      ymax=regr.mse_mean+regr.mse_sd,
      group=algorithm,
      fill=algorithm),
      help=paste("Mean plus or minus one standard deviation, over", n.folds, "cross-validation folds."),
      color=NA,
      alpha=0.5,
      data=reg.bench.wide)+
    geom_line(aes(
      train_size, regr.mse_mean,
      group=algorithm),
      help=paste("Mean over", n.folds, "cross-validation folds."),
      color="grey",
      showSelected="algorithm",
      data=reg.bench.wide)+
    scale_size_manual(values=algo.sizes)+
    scale_fill_manual(values=algo.colors)+
    scale_color_manual(values=c(
      significant="black",
      "not significant"=NA))+
    geom_point(aes(
      train_size, regr.mse_mean,
      color=difference,
      size=algorithm,
      fill=algorithm),
      help=paste("Mean over", n.folds, "cross-validation folds."),
      data=reg.bench.join)+
    geom_segment(aes(
      train_size, 10^mean.rpart,
      key=1,
      xend=train_size, yend=10^mean.featureless),
      showSelected="signal_difficulty_Ntrain",
      help="Grey segment shows difference between rpart and featureless.",
      size=3,
      alpha=0.5,
      data=test.proposed)+
    geom_text(aes(
      train_size, 10^pmax(mean.rpart,mean.featureless)*1.1,
      key=1,
      hjust=ifelse(train_size<mid.x, 0, 1),
      label=fcase(
        p.value<0.0001, "p<0.0001",
        is.nan(p.value), "Diff=0",
        default=sprintf("p=%.4f", p.value))),
      showSelected="signal_difficulty_Ntrain",
      help="P-value in unpaired two-sided T-test for difference between rpart and featureless.",
      data=test.proposed)+
    geom_rect(aes(
      xmin=xmin, xmax=xmax,
      ymin=0, ymax=Inf),
      alpha=0.1,
      help="Grey rect shows current selection of signal, difficulty, number of train samples.",
      fill="black",
      color=NA,
      clickSelects="signal_difficulty_Ntrain",
      data=test.proposed)+
    scale_y_log10(
      "Mean Squared Error (log scale)"
    )+
    scale_x_log10(
      "Ntrain = Number of samples in train set (log scale)")+
    facet_grid(difficulty ~ ., scales="free"),
  scatter=ggplot()+
    ggtitle("MSE for selected")+
    theme_bw()+
    theme_animint(width=300, height=300)+
    theme(legend.position="none")+
    coord_equal(xlim=mse.limits, ylim=mse.limits)+
    scale_x_log10(
      "featureless (log scale)",
      breaks=mse.breaks)+
    scale_y_log10(
      "rpart (log scale)",
      breaks=mse.breaks)+
    geom_abline(aes(
      slope=slope, intercept=intercept),
      help="Diagonal line represents equal prediction error for rpart and featureless.",
      color="grey50",
      data=data.table(slope=1, intercept=0))+
    geom_segment(aes(
      x, y, xend=xend, yend=yend, color=algorithm),
      data=rbind(
        data.table(x=0, y=0, xend=0, yend=Inf, algorithm="rpart"),
        data.table(x=0, y=0, xend=Inf, yend=0, algorithm="featureless")),
      help="Segments show colors corresponding to each algorithm: blue=rpart and red=featureless.",
      alpha=0.5,
      showSelected="algorithm",
      size=5)+
    scale_color_manual(values=algo.colors)+
    geom_point(aes(
      10^featureless, 10^rpart,
      key=test.fold,
      tooltip=sprintf(
        "fold %d featureless=%.3f rpart=%.3f", test.fold, featureless, rpart)),
      showSelected="signal_difficulty_Ntrain",
      help="One dot for each cross-validation fold.",
      clickSelects="test.fold",
      size=5,
      alpha=0.7,
      data=reg.bench.test),
  details=ggplot()+
    ggtitle("MSE for selected")+
    theme_bw()+
    theme_animint(width=1000, height=150)+
    theme(legend.position="none")+
    scale_y_discrete("Algo")+
    scale_x_log10(
      "Mean Squared Error (log scale)",
      limits=mse.limits,
      breaks=mse.breaks)+
    scale_color_manual(values=algo.colors)+
    geom_point(aes(
      regr.mse, algorithm,
      key=paste0(algorithm, test.fold),
      color=algorithm,
      tooltip=sprintf(
        "%s fold %d MSE=%.3f", algorithm, test.fold, regr.mse)),
      showSelected=c("algorithm","signal_difficulty_Ntrain"),
      help="One dot for each cross-validation fold and algorithm.",
      clickSelects="test.fold",
      alpha=0.7,
      size=5,
      data=reg.bench.score),
  pred=ggplot()+
    ggtitle("Predictions for selected train/test split")+
    theme_bw()+
    theme_animint(height=300, width=900)+
    geom_point(aes(
      x, y,
      key=row_id),
      showSelected=c("signal_difficulty_Ntrain","test.fold"),
      size=3,
      help="One dot for each sample in test set (left) and train set (right).",
      fill="white",
      color=data.color,
      data=Tpred(point.dt))+
    scale_x_continuous(
      "x = feature/input",
      labels=Tlabels,
      breaks=Tbreaks)+
    scale_y_continuous("y = label/output")+
    geom_line(aes(
      x, y,
      key=Set,
      group=Set,
      color=algorithm,
      size=algorithm),
      help="Black curve shows ideal prediction function, used to generate data.",
      showSelected=c("signal_difficulty_Ntrain"),
      data=Tpred(signal.dt))+
    geom_line(aes(
      x, y,
      key=paste(algorithm,Set),
      color=algorithm,
      size=algorithm,
      group=paste(algorithm,Set)),
      help="Blue and red curves show learned prediction functions.",
      showSelected=c("signal_difficulty_Ntrain","test.fold"),
      data=Tpred(pred.dt))+
    geom_text(aes(
      x, ifelse(Set=="unused", unused.y.text, Set.y.text),
      hjust=ifelse(Set=="unused", 0.5, 0),
      key=Set,
      label=paste0(Set," set N=",N)),
      data=Tpred(data.sizes[, x := 0]),
      help="Text shows number of samples in each set.",
      color=data.color,
      showSelected="signal_difficulty_Ntrain")+
    scale_size_manual(values=algo.sizes)+
    scale_color_manual(values=algo.colors),
  out.dir="cv-noise-samples",
  source="https://github.com/tdhock/2024-08-ift603-712/blob/main/cv-subsets-small-large-noise.R",
  duration=list(
    test.fold=1000,
    signal_difficulty_Ntrain=1000),
  first=list(
    signal_difficulty_Ntrain=paste("sin easy", max.N))
)

if(FALSE){
  animint2pages(viz, "2026-02-17-small-large-noise-samples")
}
viz

ggplot()+
  ggtitle("Select signal, difficulty, Ntrain")+
  theme_bw()+
  theme_animint(width=600, height=300)+
  geom_ribbon(aes(
    train_size,
    ymin=regr.mse_mean-regr.mse_sd,
    ymax=regr.mse_mean+regr.mse_sd,
    group=algorithm,
    fill=algorithm),
    help=paste("Mean plus or minus one standard deviation, over", n.folds, "cross-validation folds."),
    color=NA,
    alpha=0.5,
    data=reg.bench.wide)+
  geom_line(aes(
    train_size, regr.mse_mean,
    group=algorithm),
    help=paste("Mean over", n.folds, "cross-validation folds."),
    color="grey",
    showSelected="algorithm",
    data=reg.bench.wide)+
  scale_size_manual(values=algo.sizes)+
  scale_fill_manual(values=algo.colors)+
  scale_color_manual(values=c(
    significant="black",
    "not significant"=NA))+
  geom_point(aes(
    train_size, regr.mse_mean,
    color=difference,
    size=algorithm,
    fill=algorithm),
    help=paste("Mean over", n.folds, "cross-validation folds."),
    data=reg.bench.join)+
  scale_y_log10(
    "Mean Squared Error (log scale)"
  )+
  scale_x_log10(
    "Ntrain = Number of samples in train set (log scale)")+
  facet_grid(signal ~ difficulty)

set.seed(1)
sub_task_dt <- rbind(
  reg.data[difficulty=="easy"][sample(.N, 400)],
  reg.data[difficulty=="hard"][sample(.N, 200)]
)[, .(x,y,difficulty)]
sub_task <- mlr3::TaskRegr$new(
  "different", sub_task_dt, target="y")
sub_task$col_roles$subset <- "difficulty"
sub_task$col_roles$feature <- "x"

iid_task_dt <- reg.data[
  difficulty=="easy"
][sample(.N, 120)][
, Subset := rep(c("large","large","small"), l=.N)
][, .(x,y,Subset)]
iid_task <- mlr3::TaskRegr$new(
  "iid_easy", iid_task_dt, target="y")
iid_task$col_roles$subset <- "Subset"
iid_task$col_roles$feature <- "x"
table(iid_task_dt$Subset)

d_task_list <- list(iid_task, sub_task)
soakd <- mlr3resampling::ResamplingSameOtherSizesCV$new()
soakd$param_set$values$sizes <- 0
soakd$param_set$values$folds <- 10
soakd$param_set$values$seeds <- 1
(reg.bench.grid <- mlr3::benchmark_grid(
  d_task_list,
  reg.learner.list,
  soakd))

if(FALSE){
  if(require(future))plan("multisession")
}
if(require(lgr))get_logger("mlr3")$set_threshold("warn")
(reg.bench.result <- mlr3::benchmark(
  reg.bench.grid, store_models = TRUE))

(score_dt <- mlr3resampling::score(reg.bench.result)[, .(
  test.subset, test.fold,
  train.subsets, Train_subsets, groups, n.train.groups,
  algorithm, RMSE=sqrt(regr.mse), task_id)])

pvalue_downsample <- function(score_in, value.var=NULL, digits=3){
  Train_subsets <- train.subsets <- value <- value_mean <- value_sd <- . <- lo <- hi <- task_id <- algorithm <- test.subset <- same <- same_mean <- compare_mean <- hjust <- pmax_mean <- mid <- pmin_mean <- p.paired <- mid_lo <- mid_hi <- NULL
  if(is.null(value.var)){
    value.var <- grep("classif|regr", names(score_in), value=TRUE)[1]
    if(is.na(value.var)){
      stop("value.var=NULL which means to take the first column matching classif|regr, but there are none, so please pick one among: ", paste(names(score_in), collapse=", "))
    }
  }
  if(length(value.var) != 1){
    stop("value.var must be length=1")
  }
  if(!value.var %in% names(score_in)){
    stop("value.var must be a column name of score_in")
  }
  measure.possible <- c("other","all")
  measure.vars <- measure.possible[measure.possible %in% score_in$train.subsets]
  if(length(measure.vars)==0){
    stop("score_in$train.subsets does not contain 'all' or 'other' which are necessary for computing p-values")
  }
  levs.present <- c(
    "same",
    measure.vars,
    paste0(measure.vars,"-same"))
  levs.possible <- c(
    "all",
    "all-same",
    "same",
    "other-same",
    "other")
  levs <- c(
    levs.possible[levs.possible %in% levs.present],
    "")#for space above.
  score_panels <- rbind(
    score_in[n.train.groups==groups][, sample_size := "full"],
    score_in[n.train.groups==min(groups)][
    , sample_size := paste0("smallest=", min(groups))])
  score_dt <- add_algorithm(data.table(score_panels))[, let(
    Train_subsets = factor(train.subsets, levs),
    value = get(value.var)
  )]
  score_wide <- dcast(
    score_dt,
    sample_size + algorithm + test.fold ~ train.subsets)
  score_long <- melt(
    score_wide,
    measure.vars=measure.vars,
    variable.name="train.subsets")
  stats_dt <- dcast(
    score_dt,
    sample_size + n.train.groups + algorithm + Train_subsets ~ .,
    list(mean, sd, length)
  )[, let(
    lo=value_mean-value_sd,
    hi=value_mean+value_sd
  )]
  range_dt <- stats_dt[, {
    min_val <- min(lo,na.rm=TRUE)
    max_val <- max(hi,na.rm=TRUE)
    data.table(
      min_val,
      mid_lo=min_val*2/3+max_val*1/3,
      mid=(min_val+max_val)/2,
      mid_hi=min_val*1/3+max_val*2/3,
      max_val)
  }, by=sample_size]
  try.test <- function(...)tryCatch({
    t.test(...)
  }, error=function(e)list(estimate=NA_real_, p.value=NA_real_))
  pval_dt <- score_long[, {
    paired <- try.test(value, same, paired=TRUE)
    unpaired <- try.test(value, same, paired=FALSE)
    data.table(
      mean_diff=paired$estimate,
      diff_mean=diff(unpaired$estimate),
      p.paired=paired$p.value,
      p.unpaired=unpaired$p.value,
      same_mean=mean(same),
      compare_mean=mean(value),
      N=.N)
  }, by=.(
    sample_size, algorithm,
    Train_subsets=factor(paste0(train.subsets,"-same"), levs)
  )]
  pval_range <- range_dt[
    pval_dt, on=.(sample_size)
  ][, let(
    pmin_mean = pmin(same_mean, compare_mean),
    pmax_mean = pmax(same_mean, compare_mean)
  )][
  , hjust := fcase(
    pmax_mean < mid, 0,
    pmin_mean > mid, 1,
    default=0.5)
  ][, let(
    text_label = paste0(
      ifelse(
        p.paired<0.0001,
        "P<0.0001",
        sprintf("P=%.4f", p.paired))),
    text_value = fcase(
      hjust==0, pmin_mean,
      hjust==1, pmax_mean,
      default=(pmin_mean+pmax_mean)/2)
  )][]
  stats_range <- range_dt[
    stats_dt, on=.(sample_size)
  ][, let(
    hjust = fcase(
      value_mean<mid_lo, 0,
      value_mean>mid_hi, 1,
      default=0.5),
    text_label = sprintf(
      paste0("%.",digits,"f\u00B1%.",digits,"f%s"),
      value_mean, value_sd, ifelse(
        sample_size=="full", paste0(" N=", n.train.groups), "")
    )
  )][]
  structure(list(
    value.var=value.var,
    stats=stats_range,
    pvalues=pval_range), class=c("pvalue_downsample", "list"))
}
add_algorithm <- function(DT){
  algorithm <- learner_id <- NULL
  ## Above to avoid CRAN NOTE.
  if(is.null(DT[["algorithm"]]) && !is.null(DT[["learner_id"]]))
    DT[, algorithm := sub(".*[.]", "", learner_id)]
  DT
}
plot.pvalue_downsample <- function(x, ..., text.size=5, p.color="grey50", sd.seg.size=1){
  value_mean <- Train_subsets <- hi <- lo <- compare_mean <- same_mean <- hjust <- text_label <- text_value <- label_both <- NULL
  ggplot()+
    theme_bw()+
    geom_point(aes(
      value_mean,
      Train_subsets),
      shape=1,
      data=x$stats)+
    geom_segment(aes(
      hi,
      Train_subsets,
      xend=lo, yend=Train_subsets),
      size=sd.seg.size,
      data=x$stats)+
    geom_segment(aes(
      compare_mean, Train_subsets,
      xend=same_mean, yend=Train_subsets),
      color=p.color,
      data=x$pvalues)+
    geom_text(aes(
      value_mean,
      Train_subsets,
      hjust=hjust,
      label=text_label),
      size=text.size,
      vjust=-0.5,
      data=x$stats)+
    geom_text(aes(
      text_value, Train_subsets,
      label=text_label,
      hjust=hjust),
      color=p.color,
      size=text.size,
      vjust=-0.5,
      data=x$pvalues)+
    facet_grid(
      algorithm ~ sample_size,
      #labeller=label_both,
      scales="free")+
    scale_x_continuous(
      x$value.var)+
    scale_y_discrete(
      "Train subsets",
      drop=FALSE)
}

saveplot <- function(tid, tsub){
  pds <- pvalue_downsample(score_dt[
    algorithm=="rpart" & task_id==tid & test.subset==tsub
  ], "RMSE")
  out.png <- sprintf("cv-subsets-small-large-noise-%s-%s.png", tid, tsub)
  png(out.png, width=7, height=2.5, units="in", res=200)
  gg <- plot(pds, text.size=9)+
    theme_bw()+
    scale_x_continuous("RMSE of rpart decision tree (mean±SD over 10 fold-CV)")+
    facet_grid(. ~ sample_size, labeller=label_both)+
    ggtitle(paste0("Subsets are: ", tid, ", predict on subset: ", tsub))
  print(gg)
  dev.off()
}

myplot <- function(tid, tsub)data.table(tid,tsub)
out_dt <- rbind(
  myplot("different", "hard"),
  myplot("different", "easy"),
  myplot("iid_easy", "large"),
  myplot("iid_easy", "small"))
for(out_i in 1:nrow(out_dt)){
  out_row <- out_dt[out_i]
  out_row[, saveplot(tid, tsub)]
}
