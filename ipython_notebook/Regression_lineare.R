N <- 300
P <- 20
set.seed(1)
X.train <- matrix(runif(N*P),N,P)
norm01 <- function(x)(x-min(x))/(max(x)-min(x))
y.train <- norm01(5*X.train[,1]-2*X.train[,2]^2+3*X.train[,3]*X.train[,4]+rnorm(N))
N.folds <- 3
library(data.table)
fold.vec <- rep(1:3, l=N)
colnames(X.train) <- paste0("x", 1:P)
out.dt <- data.table(y=y.train, X.train)
fwrite(out.dt,"Regression_lineare.csv")
valid.fold <- 1
set.vec <- ifelse(fold.vec==valid.fold, "validation", "subtrain")
set.list <- list()
for(set.name in unique(set.vec)){
  is.set <- set.vec==set.name
  set.list[[set.name]] <- list(
    X=X.train[is.set,],
    y=y.train[is.set])
}



weight.vec <- rep(0,P)
max.iterations <- 1000
step.size <- 0.0001#too small
step.size <- 0.01#too big
step.size <- 0.001#good
err.dt.list <- list()
for(iteration in 1:max.iterations){
  grad.vec <- with(set.list$subtrain, t(X) %*% (X %*% weight.vec - y))
  weight.vec <- weight.vec - step.size * grad.vec
  err.dt.list[[iteration]] <- data.table(
    set=set.vec, L2err=as.numeric((X.train %*% weight.vec - y.train)^2)
  )[, .(iteration, mean.squared.error=mean(L2err)), by=set]
}
(err.dt <- rbindlist(err.dt.list))
min.dt <- err.dt[, .SD[which.min(mean.squared.error)], by=set]
library(ggplot2)
ggplot()+
  geom_line(aes(
    iteration, mean.squared.error, color=set),
    data=err.dt)+
  geom_point(aes(
    iteration, mean.squared.error, color=set),
    shape=21,
    fill="white",
    data=min.dt)+
  scale_y_log10()

ggplot()+
  facet_grid(set ~ ., labeller=label_both, scales="free")+
  geom_line(aes(
    iteration, mean.squared.error, color=set),
    data=err.dt[iteration>150])+
  geom_point(aes(
    iteration, mean.squared.error, color=set),
    shape=21,
    fill="white",
    data=min.dt)+
  scale_y_log10()
