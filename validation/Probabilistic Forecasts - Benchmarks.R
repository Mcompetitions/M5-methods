#Load libraries
library(zoo)
library(randomForest)
library(RSNNS)
library(foreach)
library(smooth)
library(doSNOW)
library(plyr)
library(forecast)

#Read data
sales <- read.csv("sales_train_validation.csv", stringsAsFactors = F)
calendar <- read.csv("calendar.csv", stringsAsFactors = F) ; calendar$date <- as.Date(calendar$date)
prices <- read.csv("sell_prices.csv", stringsAsFactors = F)
prices <- merge(prices, sales[,c(1:5)], by=c("store_id", "item_id"), all.x=T)

#This function returns some basic statistics about the forecasted series
statistics <- function(tsid){
  
  input <- time_series_b[[tsid]]
  
  lngth <- length(input$x)
  D <- demand(input$x)
  ADI <- mean(intervals(input$x))
  CV2 <- (sd(D)/mean(D))^2
  Min <- min(input$x)
  Low25 <- as.numeric(quantile(input$x,0.25))
  Mean <- mean(input$x)
  Median <- median(input$x)
  Up25 <- as.numeric(quantile(input$x,0.75))
  Max <- max(input$x)
  pz <- length(input$x[input$x==0])/lngth
  
  if (ADI > (4/3)){
    if (CV2 > 0.5){
      Type <- "Lumpy"
    }else{
      Type <- "Intermittent"
    }
  }else{
    if (CV2 > 0.5){
      Type <- "Erratic"
    }else{
      Type <- "Smooth"
    }
  }
  
  Agg_Level_1=input$Agg_Level_1
  Agg_Level_2=input$Agg_Level_2
  Hlevel=input$Hlevel
  
  matrix_s <- data.frame(tsid, Agg_Level_1, Agg_Level_2,
                         lngth, ADI, CV2, pz, Type, Min, Low25, Mean, Median, Up25, Max)
  return(matrix_s)
  
}
#These functions supoort the main forecasting functions
intervals <- function(x){
  y<-c()
  k<-1
  counter<-0
  for (tmp in (1:length(x))){
    if(x[tmp]==0){
      counter<-counter+1
    }else{
      k<-k+1
      y[k]<-counter
      counter<-1
    }
  }
  y<-y[y>0]
  y[is.na(y)]<-1
  y
}
demand <- function(x){
  y<-x[x!=0]
  y
}
#Use the benchmarks to forecast the M5 series
benchmarks_f <- function(x, fh){
  
  input <- x$x
  
  #Drop the periods where the item wasn't active
  start_period <- data.frame(input, c(1:length(input)))
  start_period <- min(start_period[start_period$input>0,2])
  input <- input[start_period:length(input)]
  
  #Estimate forecasts
  
  # SES
  sesf <- ses(ts(input, frequency = 7), h = fh, level = c(50, 67, 95, 99))
  sesf <- data.frame(cbind(sesf$lower[,4], sesf$lower[,3], sesf$lower[,2], sesf$lower[,1],
                           sesf$mean,
                           sesf$upper[,1], sesf$upper[,2], sesf$upper[,3],sesf$upper[,4]))
  colnames(sesf) <- paste0("q",qlist)
  for (qid in 1:ncol(sesf)){
    for (fhid in 1:nrow(sesf)){
      sesf[fhid,qid] <- max(0,sesf[fhid,qid])
    }
  }
  sesf <- data.frame(as.numeric(unlist(sesf)), 
                     rep(c(1:fh),9), 
                     unlist(lapply(c(1:9), function(x) rep(qlist[x],fh))))
  colnames(sesf) <- c("SES", "fh", "quantile")
  
  # Naive
  naivef <- naive(ts(input, frequency = 7), h = fh, level = c(50, 67, 95, 99))
  naivef <- data.frame(cbind(naivef$lower[,4], naivef$lower[,3], naivef$lower[,2], naivef$lower[,1],
                             naivef$mean,
                             naivef$upper[,1], naivef$upper[,2], naivef$upper[,3], naivef$upper[,4]))
  colnames(naivef) <- paste0("q",qlist)
  for (qid in 1:ncol(naivef)){
    for (fhid in 1:nrow(naivef)){
      naivef[fhid,qid] <- max(0,naivef[fhid,qid])
    }
  }
  naivef <- data.frame(as.numeric(unlist(naivef)), 
                       rep(c(1:fh),9), 
                       unlist(lapply(c(1:9), function(x) rep(qlist[x],fh))))
  colnames(naivef) <- c("Naive", "fh", "quantile")
  
  # sNaive
  snaivef <- snaive(ts(input, frequency = 7), h = fh, level = c(50, 67, 95, 99))
  snaivef <- data.frame(cbind(snaivef$lower[,4], snaivef$lower[,3], snaivef$lower[,2], snaivef$lower[,1],
                              snaivef$mean,
                              snaivef$upper[,1], snaivef$upper[,2], snaivef$upper[,3], snaivef$upper[,4]))
  colnames(snaivef) <- paste0("q",qlist)
  for (qid in 1:ncol(snaivef)){
    for (fhid in 1:nrow(snaivef)){
      snaivef[fhid,qid] <- max(0,snaivef[fhid,qid])
    }
  }
  snaivef <- data.frame(as.numeric(unlist(snaivef)), 
                        rep(c(1:fh),9), 
                        unlist(lapply(c(1:9), function(x) rep(qlist[x],fh))))
  colnames(snaivef) <- c("sNaive", "fh", "quantile")
  
  # ETS
  etsf <- forecast(ets(ts(input, frequency = 7)), h = fh, level = c(50, 67, 95, 99))
  etsf <- data.frame(cbind(etsf$lower[,4], etsf$lower[,3], etsf$lower[,2], etsf$lower[,1],
                           etsf$mean,
                           etsf$upper[,1], etsf$upper[,2], etsf$upper[,3], etsf$upper[,4]))
  colnames(etsf) <- paste0("q",qlist)
  for (qid in 1:ncol(etsf)){
    for (fhid in 1:nrow(etsf)){
      etsf[fhid,qid] <- max(0,etsf[fhid,qid])
    }
  }
  etsf <- data.frame(as.numeric(unlist(etsf)), 
                     rep(c(1:fh),9), 
                     unlist(lapply(c(1:9), function(x) rep(qlist[x],fh))))
  colnames(etsf) <- c("ETS", "fh", "quantile")
  
  # ARIMA
  arimaf <- forecast(auto.arima(ts(input, frequency = 7)), h = fh, level = c(50, 67, 95, 99))
  arimaf <- data.frame(cbind(arimaf$lower[,4], arimaf$lower[,3], arimaf$lower[,2], arimaf$lower[,1],
                             arimaf$mean,
                             arimaf$upper[,1], arimaf$upper[,2], arimaf$upper[,3], arimaf$upper[,4]))
  colnames(arimaf) <- paste0("q",qlist)
  for (qid in 1:ncol(arimaf)){
    for (fhid in 1:nrow(arimaf)){
      arimaf[fhid,qid] <- max(0,arimaf[fhid,qid])
    }
  }
  arimaf <- data.frame(as.numeric(unlist(arimaf)), 
                       rep(c(1:fh),9), 
                       unlist(lapply(c(1:9), function(x) rep(qlist[x],fh))))
  colnames(arimaf) <- c("ARIMA", "fh", "quantile")
  
  # Kernel 
  quants <- as.numeric(quantile(input,qlist))
  kernelf <- data.frame(cbind(rep(quants[1],28), rep(quants[2],28), rep(quants[3],28),
                              rep(quants[4],28), rep(quants[5],28), rep(quants[6],28),
                              rep(quants[7],28), rep(quants[8],28), rep(quants[9],28)))
  colnames(kernelf) <- paste0("q",qlist)
  kernelf <- data.frame(as.numeric(unlist(kernelf)), 
                        rep(c(1:fh),9), 
                        unlist(lapply(c(1:9), function(x) rep(qlist[x],fh))))
  colnames(kernelf) <- c("Kernel", "fh", "quantile")
  
  Methods <- merge(naivef, snaivef, by=c("fh", "quantile"))
  Methods <- merge(Methods, sesf, by=c("fh", "quantile"))
  Methods <- merge(Methods, etsf, by=c("fh", "quantile"))
  Methods <- merge(Methods, arimaf, by=c("fh", "quantile"))
  Methods <- merge(Methods, kernelf, by=c("fh", "quantile"))
  
  Methods$Agg_Level_1	<- x$Agg_Level_1
  Methods$Agg_Level_2 <- x$Agg_Level_2
  
  return(Methods)
}

## Preparate the series to be predicted (bottom level)
time_series_b <- NULL

#Level 1
ex_sales <- sales
sales_train <- ex_sales[,6:ncol(ex_sales)]
input <- as.numeric(colSums(sales_train, na.rm = TRUE))
time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                    Agg_Level_1="Total", 
                                                    Agg_Level_2="X",
                                                    Hlevel=1))

#Level 2
for (tsid in 1:length(unique(sales$state_id))){
  ex_sales <- sales[sales$state_id==unique(sales$state_id)[tsid],]
  sales_train <- ex_sales[,6:ncol(ex_sales)]
  sales_train <- colSums(sales_train, na.rm = TRUE)
  sales_train <- data.frame(sales_train, c(1:length(sales_train)))
  starting_period <- min(sales_train[sales_train$sales_train>0,2])
  input <- sales_train$sales_train[starting_period:nrow(sales_train)]
  time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                      Agg_Level_1=unique(sales$state_id)[tsid], 
                                                      Agg_Level_2="X",
                                                      Hlevel=2))
}

#Level 3
for (tsid in 1:length(unique(sales$store_id))){
  ex_sales <- sales[sales$store_id==unique(sales$store_id)[tsid],]
  sales_train <- ex_sales[,6:ncol(ex_sales)]
  sales_train <- colSums(sales_train, na.rm = TRUE)
  sales_train <- data.frame(sales_train, c(1:length(sales_train)))
  starting_period <- min(sales_train[sales_train$sales_train>0,2])
  input <- sales_train$sales_train[starting_period:nrow(sales_train)]
  time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                      Agg_Level_1=unique(sales$store_id)[tsid], 
                                                      Agg_Level_2="X",
                                                      Hlevel=3))
}

#Level 4
for (tsid in 1:length(unique(sales$cat_id))){
  ex_sales <- sales[sales$cat_id==unique(sales$cat_id)[tsid],]
  sales_train <- ex_sales[,6:ncol(ex_sales)]
  sales_train <- colSums(sales_train, na.rm = TRUE)
  sales_train <- data.frame(sales_train, c(1:length(sales_train)))
  starting_period <- min(sales_train[sales_train$sales_train>0,2])
  input <- sales_train$sales_train[starting_period:nrow(sales_train)]
  time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                      Agg_Level_1=unique(sales$cat_id)[tsid], 
                                                      Agg_Level_2="X",
                                                      Hlevel=4))
}

#Level 5
for (tsid in 1:length(unique(sales$dept_id))){
  ex_sales <- sales[sales$dept_id==unique(sales$dept_id)[tsid],]
  sales_train <- ex_sales[,6:ncol(ex_sales)]
  sales_train <- colSums(sales_train, na.rm = TRUE)
  sales_train <- data.frame(sales_train, c(1:length(sales_train)))
  starting_period <- min(sales_train[sales_train$sales_train>0,2])
  input <- sales_train$sales_train[starting_period:nrow(sales_train)]
  time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                      Agg_Level_1=unique(sales$dept_id)[tsid], 
                                                      Agg_Level_2="X",
                                                      Hlevel=5))
}

#Level 6
for (tsid1 in 1:length(unique(sales$state_id))){
  for (tsid2 in 1:length(unique(sales$cat_id))){
    ex_sales <- sales[(sales$state_id==unique(sales$state_id)[tsid1])&
                        (sales$cat_id==unique(sales$cat_id)[tsid2]),]
    sales_train <- ex_sales[,6:ncol(ex_sales)]
    sales_train <- colSums(sales_train, na.rm = TRUE)
    sales_train <- data.frame(sales_train, c(1:length(sales_train)))
    starting_period <- min(sales_train[sales_train$sales_train>0,2])
    input <- sales_train$sales_train[starting_period:nrow(sales_train)]
    time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                        Agg_Level_1=unique(sales$state_id)[tsid1], 
                                                        Agg_Level_2=unique(sales$cat_id)[tsid2],
                                                        Hlevel=6))
  }
}

#Level 7
for (tsid1 in 1:length(unique(sales$state_id))){
  for (tsid2 in 1:length(unique(sales$dept_id))){
    ex_sales <- sales[(sales$state_id==unique(sales$state_id)[tsid1])&
                        (sales$dept_id==unique(sales$dept_id)[tsid2]),]
    sales_train <- ex_sales[,6:ncol(ex_sales)]
    sales_train <- colSums(sales_train, na.rm = TRUE)
    sales_train <- data.frame(sales_train, c(1:length(sales_train)))
    starting_period <- min(sales_train[sales_train$sales_train>0,2])
    input <- sales_train$sales_train[starting_period:nrow(sales_train)]
    time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                        Agg_Level_1=unique(sales$state_id)[tsid1], 
                                                        Agg_Level_2=unique(sales$dept_id)[tsid2],
                                                        Hlevel=7))
  }
}

#Level 8
for (tsid1 in 1:length(unique(sales$store_id))){
  for (tsid2 in 1:length(unique(sales$cat_id))){
    ex_sales <- sales[(sales$store_id==unique(sales$store_id)[tsid1])&
                        (sales$cat_id==unique(sales$cat_id)[tsid2]),]
    sales_train <- ex_sales[,6:ncol(ex_sales)]
    sales_train <- colSums(sales_train, na.rm = TRUE)
    sales_train <- data.frame(sales_train, c(1:length(sales_train)))
    starting_period <- min(sales_train[sales_train$sales_train>0,2])
    input <- sales_train$sales_train[starting_period:nrow(sales_train)]
    time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                        Agg_Level_1=unique(sales$store_id)[tsid1], 
                                                        Agg_Level_2=unique(sales$cat_id)[tsid2],
                                                        Hlevel=8))
  }
}

#Level 9
for (tsid1 in 1:length(unique(sales$store_id))){
  for (tsid2 in 1:length(unique(sales$dept_id))){
    ex_sales <- sales[(sales$store_id==unique(sales$store_id)[tsid1])&
                        (sales$dept_id==unique(sales$dept_id)[tsid2]),]
    sales_train <- ex_sales[,6:ncol(ex_sales)]
    sales_train <- colSums(sales_train, na.rm = TRUE)
    sales_train <- data.frame(sales_train, c(1:length(sales_train)))
    starting_period <- min(sales_train[sales_train$sales_train>0,2])
    input <- sales_train$sales_train[starting_period:nrow(sales_train)]
    time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                        Agg_Level_1=unique(sales$store_id)[tsid1], 
                                                        Agg_Level_2=unique(sales$dept_id)[tsid2],
                                                        Hlevel=9))
  }
}

#Level 10
for (tsid in 1:length(unique(sales$item_id))){
  ex_sales <- sales[sales$item_id==unique(sales$item_id)[tsid],]
  sales_train <- ex_sales[,6:ncol(ex_sales)]
  sales_train <- colSums(sales_train, na.rm = TRUE)
  sales_train <- data.frame(sales_train, c(1:length(sales_train)))
  starting_period <- min(sales_train[sales_train$sales_train>0,2])
  input <- sales_train$sales_train[starting_period:nrow(sales_train)]
  time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                      Agg_Level_1=unique(sales$item_id)[tsid], 
                                                      Agg_Level_2="X",
                                                      Hlevel=10))
}

#Level 11
for (tsid1 in 1:length(unique(sales$state_id))){
  for (tsid2 in 1:length(unique(sales$item_id))){
    ex_sales <- sales[(sales$state_id==unique(sales$state_id)[tsid1])&
                        (sales$item_id==unique(sales$item_id)[tsid2]),]
    sales_train <- ex_sales[,6:ncol(ex_sales)]
    sales_train <- colSums(sales_train, na.rm = TRUE)
    sales_train <- data.frame(sales_train, c(1:length(sales_train)))
    starting_period <- min(sales_train[sales_train$sales_train>0,2])
    input <- sales_train$sales_train[starting_period:nrow(sales_train)]
    time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                        Agg_Level_1=unique(sales$state_id)[tsid1], 
                                                        Agg_Level_2=unique(sales$item_id)[tsid2],
                                                        Hlevel=11))
  }
}

#Level 12
for (tsid in 1:nrow(sales)){
  ex_sales <- sales[tsid,]
  sales_train <- as.numeric(ex_sales[,6:ncol(ex_sales)]) 
  sales_train <- data.frame(sales_train, c(1:length(sales_train)))
  starting_period <- min(sales_train[sales_train$sales_train>0,2])
  input <- sales_train$sales_train[starting_period:nrow(sales_train)]
  time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                      Agg_Level_1=ex_sales$item_id, 
                                                      Agg_Level_2=ex_sales$store_id,
                                                      Hlevel=12))
}

rm(input, sales_train, ex_sales, starting_period, tsid, tsid1, tsid2)
#############################################################################################################

cl = registerDoSNOW(makeCluster(8, type = "SOCK"))
##  Estimate some basic statistics and the dollar sales used for weighting  
stat_total <- foreach(tsi=1:length(time_series_b), .combine='rbind') %dopar% statistics(tsi)
## Estimate forecasts for benchmark methods      
qlist <- c(0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)
b_names <- c("Naive", "sNaive", "SES", "ETS", "ARIMA", "Kernel")
frc_total <- foreach(tsi=1:length(time_series_b), .combine='rbind', 
                     .packages=c('zoo','forecast')) %dopar% benchmarks_f(time_series_b[[tsi]], 28)

## Evaluate forecasts 
sales_out <- read.csv("sales_test_validation.csv", stringsAsFactors = F)
weights <- read.csv("weights_validation.csv", stringsAsFactors = F) #based on last 28 days to compute WRMSSE

#Create test series based on the outsample data
time_series_b_out <- NULL
#Level 1
ex_sales_out <- sales_out
sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
input <- as.numeric(colSums(sales_train, na.rm = TRUE))
time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                            Agg_Level_1="Total", 
                                                            Agg_Level_2="X",
                                                            Hlevel=1))


#Level 2
for (tsid in 1:length(unique(sales_out$state_id))){
  ex_sales_out <- sales_out[sales_out$state_id==unique(sales_out$state_id)[tsid],]
  sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
  sales_train <- colSums(sales_train, na.rm = TRUE)
  input <- as.numeric(sales_train)
  time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                              Agg_Level_1=unique(sales_out$state_id)[tsid], 
                                                              Agg_Level_2="X",
                                                              Hlevel=2))
}

#Level 3
for (tsid in 1:length(unique(sales_out$store_id))){
  ex_sales_out <- sales_out[sales_out$store_id==unique(sales_out$store_id)[tsid],]
  sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
  sales_train <- colSums(sales_train, na.rm = TRUE)
  input <- as.numeric(sales_train)
  time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                              Agg_Level_1=unique(sales_out$store_id)[tsid], 
                                                              Agg_Level_2="X",
                                                              Hlevel=3))
}

#Level 4
for (tsid in 1:length(unique(sales_out$cat_id))){
  ex_sales_out <- sales_out[sales_out$cat_id==unique(sales_out$cat_id)[tsid],]
  sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
  sales_train <- colSums(sales_train, na.rm = TRUE)
  input <- as.numeric(sales_train)
  time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                              Agg_Level_1=unique(sales_out$cat_id)[tsid], 
                                                              Agg_Level_2="X",
                                                              Hlevel=4))
}

#Level 5
for (tsid in 1:length(unique(sales_out$dept_id))){
  ex_sales_out <- sales_out[sales_out$dept_id==unique(sales_out$dept_id)[tsid],]
  sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
  sales_train <- colSums(sales_train, na.rm = TRUE)
  input <- as.numeric(sales_train)
  time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                              Agg_Level_1=unique(sales_out$dept_id)[tsid], 
                                                              Agg_Level_2="X",
                                                              Hlevel=5))
}

#Level 6
for (tsid1 in 1:length(unique(sales_out$state_id))){
  for (tsid2 in 1:length(unique(sales_out$cat_id))){
    ex_sales_out <- sales_out[(sales_out$state_id==unique(sales_out$state_id)[tsid1])&
                                (sales_out$cat_id==unique(sales_out$cat_id)[tsid2]),]
    sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
    sales_train <- colSums(sales_train, na.rm = TRUE)
    input <- as.numeric(sales_train)
    time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                                Agg_Level_1=unique(sales_out$state_id)[tsid1], 
                                                                Agg_Level_2=unique(sales_out$cat_id)[tsid2],
                                                                Hlevel=6))
  }
}

#Level 7
for (tsid1 in 1:length(unique(sales_out$state_id))){
  for (tsid2 in 1:length(unique(sales_out$dept_id))){
    ex_sales_out <- sales_out[(sales_out$state_id==unique(sales_out$state_id)[tsid1])&
                                (sales_out$dept_id==unique(sales_out$dept_id)[tsid2]),]
    sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
    sales_train <- colSums(sales_train, na.rm = TRUE)
    input <- as.numeric(sales_train)
    time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                                Agg_Level_1=unique(sales_out$state_id)[tsid1], 
                                                                Agg_Level_2=unique(sales_out$dept_id)[tsid2],
                                                                Hlevel=7))
  }
}

#Level 8
for (tsid1 in 1:length(unique(sales_out$store_id))){
  for (tsid2 in 1:length(unique(sales_out$cat_id))){
    ex_sales_out <- sales_out[(sales_out$store_id==unique(sales_out$store_id)[tsid1])&
                                (sales_out$cat_id==unique(sales_out$cat_id)[tsid2]),]
    sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
    sales_train <- colSums(sales_train, na.rm = TRUE)
    input <- as.numeric(sales_train)
    time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                                Agg_Level_1=unique(sales_out$store_id)[tsid1], 
                                                                Agg_Level_2=unique(sales_out$cat_id)[tsid2],
                                                                Hlevel=8))
  }
}

#Level 9
for (tsid1 in 1:length(unique(sales_out$store_id))){
  for (tsid2 in 1:length(unique(sales_out$dept_id))){
    ex_sales_out <- sales_out[(sales_out$store_id==unique(sales_out$store_id)[tsid1])&
                                (sales_out$dept_id==unique(sales_out$dept_id)[tsid2]),]
    sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
    sales_train <- colSums(sales_train, na.rm = TRUE)
    input <- as.numeric(sales_train)
    time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                                Agg_Level_1=unique(sales_out$store_id)[tsid1], 
                                                                Agg_Level_2=unique(sales_out$dept_id)[tsid2],
                                                                Hlevel=9))
  }
}

#Level 10
for (tsid in 1:length(unique(sales_out$item_id))){
  ex_sales_out <- sales_out[sales_out$item_id==unique(sales_out$item_id)[tsid],]
  sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
  sales_train <- colSums(sales_train, na.rm = TRUE)
  input <- as.numeric(sales_train)
  time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                              Agg_Level_1=unique(sales_out$item_id)[tsid], 
                                                              Agg_Level_2="X",
                                                              Hlevel=10))
}

#Level 11
for (tsid1 in 1:length(unique(sales_out$state_id))){
  for (tsid2 in 1:length(unique(sales_out$item_id))){
    ex_sales_out <- sales_out[(sales_out$state_id==unique(sales_out$state_id)[tsid1])&
                                (sales_out$item_id==unique(sales_out$item_id)[tsid2]),]
    sales_train <- ex_sales_out[,6:ncol(ex_sales_out)]
    sales_train <- colSums(sales_train, na.rm = TRUE)
    input <- as.numeric(sales_train)
    time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                                Agg_Level_1=unique(sales_out$state_id)[tsid1], 
                                                                Agg_Level_2=unique(sales_out$item_id)[tsid2],
                                                                Hlevel=11))
  }
}

#Level 12
for (tsid in 1:nrow(sales_out)){
  ex_sales_out <- sales_out[tsid,]
  sales_train <- as.numeric(ex_sales_out[,6:ncol(ex_sales_out)]) 
  input <- as.numeric(sales_train)
  time_series_b_out[length(time_series_b_out)+1] <- list(list(x=input, 
                                                              Agg_Level_1=ex_sales_out$item_id, 
                                                              Agg_Level_2=ex_sales_out$store_id,
                                                              Hlevel=12))
}

final_scores <- NULL
for (bid in b_names){
  score_total <- c()
  
  for (i in 1:length(time_series_b)){
    
    insample <- time_series_b[[i]]
    outsample <- time_series_b_out[[i]]
    
    for_temp <- frc_total[(frc_total$Agg_Level_1==insample$Agg_Level_1)&
                            (frc_total$Agg_Level_2==insample$Agg_Level_2),c("fh","quantile",bid)]
    
    wi <- weights[(weights$Agg_Level_1==outsample$Agg_Level_1)&
                    (weights$Agg_Level_2==outsample$Agg_Level_2),]$Weight
    
    scorej <- 0
    for (j in 1:length(qlist)){
      for_temp_j <- for_temp[for_temp$quantile==qlist[j],]
      for_temp_j$fh <- as.numeric(for_temp_j$fh)
      for_temp_j <- for_temp_j[order(for_temp_j$fh),] ; row.names(for_temp_j) <- NULL
      for_temp_j <- for_temp_j[,3]
      temp_score <- 0
      for (k in 1:length(for_temp_j)){
        if (outsample$x[k]>=for_temp_j[k]){
          temp_score <- temp_score + qlist[j]*(outsample$x[k]-for_temp_j[k])
        }else{
          temp_score <- temp_score + (1-qlist[j])*(for_temp_j[k]-outsample$x[k])
        }
      }
      temp_score <- (temp_score/length(for_temp_j))/mean(abs(diff(insample$x)))
      scorej <- scorej + temp_score
    }
    score_total <- rbind(score_total, data.frame((scorej/length(qlist))*wi,outsample$Agg_Level_1,outsample$Agg_Level_2, outsample$Hlevel)) 
  }
  
  colnames(score_total) <- c("SPL", "Agg_Level_1", "Agg_Level_2", "Level")
  SPL_score <- ddply(score_total[,c("SPL", "Level")], .(Level), colwise(sum))
  colnames(SPL_score)[2] <- bid
  if (bid=="Naive"){
    final_scores <- SPL_score
  }else{
    final_scores <- merge(final_scores, SPL_score, by="Level")
  }
}

## Export forecasts on the template format
template <- read.csv("sample_submission.csv", stringsAsFactors = F)

for (mid in b_names){
  
  submission <- frc_total[,c("Agg_Level_1", "Agg_Level_2", mid, "quantile", "fh")]
  submission$quantile <- as.character(submission$quantile)
  submission[submission$quantile=="0.25",]$quantile <- "0.250"
  submission[submission$quantile=="0.5",]$quantile <- "0.500"
  submission[submission$quantile=="0.75",]$quantile <- "0.750"
  submission$id <- paste0(submission$Agg_Level_1,"_",submission$Agg_Level_2,"_",submission$quantile,"_validation")
  template_temp <- template[template$id %in% submission$id,]
  submission <- submission[order(submission$fh),]
  
  sidlist <- unique(template_temp$id)
  for (sid in 1:length(sidlist)){
    template_temp[template_temp$id==sidlist[sid],2:29] <- submission[submission$id==sidlist[sid],3]
  }
  
  write.csv(template_temp, row.names = FALSE, paste0("PIs - validation/PF_", mid,".csv"))
  
}
