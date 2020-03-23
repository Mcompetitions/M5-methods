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
  
  item_id=input$item_id
  dept_id=input$dept_id
  cat_id=input$cat_id
  store_id=input$store_id 
  state_id=input$state_id
  
  ex_price <- prices[(prices$item_id==input$item_id)&(prices$store_id==input$store_id),] #Prices
  ex_calendar <- merge(calendar, ex_price, by=c("wm_yr_wk"), all.x = T) #merge with calendar
  ex_calendar <- ex_calendar[order(ex_calendar$date),] ; row.names(ex_calendar) <- NULL
  ex_dataset <- ex_calendar[,c("date", "wm_yr_wk", "sell_price")]
  
  ex_dataset$sales <- c(rep(NA, nrow(ex_dataset)-lngth-56), input$x, rep(NA, 56))
  ex_dataset <- head(tail(ex_dataset,3*28),28)
  
  
  dollar_sales <- sum(ex_dataset$sell_price*ex_dataset$sales, na.rm = T)
  
  matrix_s <- data.frame(tsid, item_id, dept_id, cat_id, store_id, state_id,
                         lngth, ADI, CV2, pz, Type, Min, Low25, Mean, Median, Up25, Max, 
                         dollar_sales)
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
recompose <- function(x,y1,y2,k){
  z1=z2<-c()
  
  tmp<-1
  for (t in (1):(length(x)-k)){
    if (x[t]==0){
      tmp<-tmp
    }else{
      tmp<-tmp+1
    }
    z1[t+1]<-y1[tmp]
    z2[t+1]<-y2[tmp]
  }
  z<-z1/z2
  head(z, length(x))
}
CreateSamples<-function(datasample,xi){
  xo<-1
  sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
  for (cid in (xi+xo):length(datasample)){
    sample[cid,]<-datasample[(cid-xi-xo+1):cid]
  }
  sample<-as.matrix(data.frame(na.omit(sample)))
  return(sample)
}
#These functions implement the M5 benchmarks
Naive <- function(x, h, type){
  frcst <- rep(tail(x,1), h)
  if (type=="seasonal"){
    frcst <- head(rep(as.numeric(tail(x,7)), h), h) 
  }
  return(frcst)
}
SexpS <- function(x, h){
  a <- optim(c(0), SES, x=x, h=1, job="train", lower = 0.1, upper = 0.3, method = "L-BFGS-B")$par
  y <- SES(a=a, x=x, h=1, job="forecast")$mean
  forecast <- rep(as.numeric(y), h)
  return(forecast)
}
SES <- function(a, x, h, job){
  y <- c()  
  y[1] <- x[1] #initialization
  
  for (t in 1:(length(x))){
    y[t+1] <- a*x[t]+(1-a)*y[t]
  }
  
  fitted <- head(y,(length(y)-1))
  forecast <- rep(tail(y,1),h)
  if (job=="train"){
    return(mean((fitted - x)^2))
  }else if (job=="fit"){
    return(fitted)
  }else{
    return(list(fitted=fitted,mean=forecast))
  }
}
MA <- function(x, h){
  mse <- c()
  for (k in 2:14){
    y <- rep(NA, k)
    for (i in (k+1):length(x)){
      y <- c(y, mean(x[(i-k):(i-1)]))
    }
    mse <- c(mse, mean((y-x)^2, na.rm = T))
  }
  k <- which.min(mse)+1
  forecast <- rep(mean(as.numeric(tail(x, k))), h)
  return(forecast)
}
Croston <- function(x, h, type){
  if (type=="classic"){
    mult <- 1 
    a1 = a2 <- 0.1 
  }else if (type=="optimized"){
    mult <- 1 
    a1 <- optim(c(0), SES, x=demand(x), h=1, job="train", lower = 0.1, upper = 0.3, method = "L-BFGS-B")$par
    a2 <- optim(c(0), SES, x=intervals(x), h=1, job="train", lower = 0.1, upper = 0.3, method = "L-BFGS-B")$par
  }else if (type=="sba"){
    mult <- 0.95
    a1 = a2 <- 0.1
  }
  yd <- SES(a=a1, x=demand(x), h=1, job="forecast")$mean
  yi <- SES(a=a2, x=intervals(x), h=1, job="forecast")$mean
  forecast <- rep(as.numeric(yd/yi), h)*mult
  return(forecast)
}
TSB <- function(x, h){
  n <- length(x)
  p <- as.numeric(x != 0)
  z <- x[x != 0]
  
  a <- c(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.8) 
  b <- c(0.01,0.02,0.03,0.05,0.1,0.2,0.3)
  MSE <- c() ; forecast <- NULL
  for (atemp in a){
    for (btemp in b){
      zfit <- vector("numeric", length(x))
      pfit <- vector("numeric", length(x))
      zfit[1] <- z[1] ; pfit[1] <- p[1]
      
      for (i in 2:n) {
        pfit[i] <- pfit[i-1] + atemp*(p[i]-pfit[i-1])
        if (p[i] == 0) {
          zfit[i] <- zfit[i-1]
        }else {
          zfit[i] <- zfit[i-1] + btemp*(x[i]-zfit[i-1])
        }
      }
      yfit <- pfit * zfit
      forecast[length(forecast)+1] <- list(rep(yfit[n], h))
      yfit <- c(NA, head(yfit, n-1))
      MSE <- c(MSE, mean((yfit-x)^2, na.rm = T) )
    }
  }
  return(forecast[[which.min(MSE)]])
}
ADIDA <- function(x, h){
  al <- round(mean(intervals(x)),0) #mean inter-demand interval
  #Aggregated series (AS)
  AS <- as.numeric(na.omit(as.numeric(rollapply(tail(x, (length(x) %/% al)*al), al, FUN=sum, by = al))))
  forecast <- rep(SexpS(AS, 1)/al, h)
  return(forecast)
}
iMAPA <- function(x, h){
  mal <- round(mean(intervals(x)),0)
  frc <- NULL
  for (al in 1:mal){
    frc <- rbind(frc, rep(SexpS(as.numeric(na.omit(as.numeric(rollapply(tail(x, (length(x) %/% al)*al), al, FUN=sum, by = al)))), 1)/al, h))
  }
  forecast <- colMeans(frc)
  return(forecast)
}
MLP_local <- function(input, fh, ni){
  
  #Scale data
  MAX <- max(input) ; MIN <- min(input)
  Sinsample <- (input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate <- CreateSamples(datasample=Sinsample,xi=ni)
  dftest <- data.frame(samplegenerate)
  colnames(dftest) <- c(paste0("X",c(1:ni)),"Y")
  train <- as.matrix(dftest[,1:(ncol(dftest)-1)])
  test <- dftest[,ncol(dftest)]
  
  #Train model
  frc_f <- NULL
  for (ssn in c(1:10)){
    
    modelMLP <- mlp(train, test, 
                    size = (2*ni), maxit = 500,initFunc = "Randomize_Weights", 
                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic", 
                    shufflePatterns = FALSE, linOut = TRUE)
    
    #Predict
    tempin <- data.frame(t(tail(Sinsample, ni)))
    MLf <- rep(as.numeric(predict(modelMLP, tempin))*(MAX-MIN)+MIN, fh)
    frc_f <- rbind(frc_f, MLf)
  }
  frc <- unlist(lapply(c(1:fh), function(x) median(frc_f[,x])))
  return(frc)
}
RF_local <- function(input, fh, ni){
  
  #Scale data
  MAX <- max(input) ; MIN <- min(input)
  Sinsample <- (input-MIN)/(MAX-MIN)
  
  #Create training sample
  samplegenerate <- CreateSamples(datasample=Sinsample,xi=ni)
  dftest <- data.frame(samplegenerate)
  colnames(dftest) <- c(paste0("X",c(1:ni)),"Y")
  
  #Train model
  modelRF <- randomForest(formula = Y ~ .,  data= dftest, ntree=500)
  
  #Predict
  tempin <- data.frame(t(tail(Sinsample, ni)))
  MLf <- as.numeric(predict(modelRF,tempin))*(MAX-MIN)+MIN
  frc <- rep(MLf, fh)
  
  return(frc)
}
#Use the benchmarks to forecast the M5 series
benchmarks_f <- function(x, fh){
  
  input <- x$x
  
  #Names of benchmarks
  fm_names <- b_names
  
  #Drop the periods where the item wasn't active
  start_period <- data.frame(input, c(1:length(input)))
  start_period <- min(start_period[start_period$input>0,2])
  input <- input[start_period:length(input)]
  
  #Estimate forecasts
  Methods <- NULL
  Methods <- cbind(Methods, Naive(input, fh, type="simple"))
  Methods <- cbind(Methods, Naive(input, fh, type="seasonal"))
  Methods <- cbind(Methods, SexpS(input, fh))
  Methods <- cbind(Methods, MA(input, fh))
  Methods <- cbind(Methods, Croston(input, fh, "classic"))
  Methods <- cbind(Methods, Croston(input, fh, "optimized"))
  Methods <- cbind(Methods, Croston(input, fh, "sba"))
  Methods <- cbind(Methods, TSB(input, fh))
  Methods <- cbind(Methods, ADIDA(input, fh))
  Methods <- cbind(Methods, iMAPA(input, fh))
  Methods <- cbind(Methods, as.numeric(es(ts(input, frequency = 7), h=fh)$forecast))
  Methods <- cbind(Methods, forecast(auto.arima(ts(input, frequency = 7)), h=fh)$mean)
  Methods <- cbind(Methods, MLP_local(input, fh, 14))
  Methods <- cbind(Methods, RF_local(input, fh, 14))
  
  #Set negatives to zero (if any)
  for (i in 1:nrow(Methods)){
    for (j in 1:ncol(Methods)){
      if (Methods[i,j]<0){ Methods[i,j]<-0  } 
    }
  }
  Methods <- data.frame(Methods)
  colnames(Methods) <- fm_names
  
  Methods$item_id <- x$item_id
  Methods$dept_id <- x$dept_id
  Methods$cat_id <- x$cat_id
  Methods$store_id <- x$store_id
  Methods$state_id <- x$state_id
  Methods$fh <- c(1:fh)
  
  return(Methods)
}
ML_Global <- function(fh){
  
  ni <- 12 ; nwindows <- 3
  
  x_train = y_train = x_test <- NULL
  Maxies = Minies <- c()
  
  #Create a sample for training
  for (i in 1:length(time_series_b)){
    input <- time_series_b[[i]]$x
    
    #Scale data
    MAX <- max(input) ; MIN <- min(input)
    Sinsample <- (input-MIN)/(MAX-MIN)
    Maxies <- c(Maxies, MAX)
    Minies <- c(Minies, MIN)
    
    #Create training sample
    samplegenerate <- CreateSamples(datasample=Sinsample,xi=ni)
    dftest <- data.frame(samplegenerate)
    colnames(dftest) <- c(paste0("X",c(1:ni)),"Y")
    train <- as.matrix(dftest[,1:(ncol(dftest)-1)])
    test <- dftest[,ncol(dftest)]
    #Select windows
    select <- sample(c(1:nrow(train)), nwindows, replace = F)
    
    temp <- data.frame(train[select,])
    temp$ADI <- stat_total[i,]$ADI
    temp$CV2 <- stat_total[i,]$CV2
    
    x_train <- rbind(x_train, temp)
    y_train <- c(y_train, test[select])
    
    temp <- data.frame(t((tail(time_series_b[[i]]$x, ni)-MIN)/(MAX-MIN)))
    temp$ADI <- stat_total[i,]$ADI
    temp$CV2 <- stat_total[i,]$CV2
    
    x_test <- rbind(x_test, temp)
  }
  colnames(x_test) <- colnames(x_train)
  x_train$ADI <- (x_train$ADI-min(x_train$ADI))/(max(x_train$ADI)-min(x_train$ADI))
  x_test$ADI <- (x_test$ADI-min(x_test$ADI))/(max(x_test$ADI)-min(x_test$ADI))
  x_train$CV2 <- (x_train$CV2-min(x_train$CV2))/(max(x_train$CV2)-min(x_train$CV2))
  x_test$CV2 <- (x_test$CV2-min(x_test$CV2))/(max(x_test$CV2)-min(x_test$CV2))
  
  #MLP
  frc_f <- NULL
  for (ssn in c(1:10)){
    modelMLP <- mlp(x_train, y_train, 
                    size = (2*ncol(x_train)), maxit = 500,initFunc = "Randomize_Weights", 
                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic", 
                    shufflePatterns = FALSE, linOut = TRUE) 
    
    frc_f <- cbind(frc_f, as.numeric(predict(modelMLP, x_test)*(Maxies-Minies)+Minies))
  }
  frc_f <- unlist(lapply(c(1:nrow(frc_f)), function(x) median(frc_f[x,])))
  MLP_g <- unlist(lapply(c(1:length(frc_f)), function(x) rep(frc_f[x], fh)))
  
  #RF
  dftest <- cbind(x_train, y_train) ; colnames(dftest)[ncol(dftest)] <- "Y"
  modelRF <- randomForest(formula = Y ~ .,  data= dftest, ntree=500)
  frc_f <- as.numeric(predict(modelRF, x_test)*(Maxies-Minies)+Minies)
  RF_g <- unlist(lapply(c(1:length(frc_f)), function(x) rep(frc_f[x], fh)))
  
  output <- data.frame(MLP_g, RF_g)
  
  return(output)
}

#############################################################################################################
#########################      Preparate the series to be predicted (bottom level)  #########################
time_series_b <- NULL
for (tsid in 1:nrow(sales)){
  #Fetch the series to be forecasted
  ex_sales <- sales[tsid,]
  #Prepare the series - keep only valid observations (first non-zero demand and after)
  sales_train <- as.numeric(ex_sales[,6:ncol(ex_sales)]) 
  sales_train <- data.frame(sales_train, c(1:length(sales_train)))
  starting_period <- min(sales_train[sales_train$sales_train>0,2])
  input <- sales_train$sales_train[starting_period:nrow(sales_train)]
  
  time_series_b[length(time_series_b)+1] <- list(list(x=input, 
                                                      item_id=ex_sales$item_id, dept_id=ex_sales$dept_id, 
                                                      cat_id=ex_sales$cat_id, store_id=ex_sales$store_id, 
                                                      state_id=ex_sales$state_id))
}
input = sales_train = ex_sales = starting_period = tsid <- NULL
#############################################################################################################

cl = registerDoSNOW(makeCluster(8, type = "SOCK"))
#############################################################################################################
##########      Estimate some basic statistics and the dollar sales used for weighting  #####################
stat_total <- foreach(tsi=1:length(time_series_b), .combine='rbind') %dopar% statistics(tsi)

save.image("Series and Stats.Rdata")

#############################################################################################################


#############################################################################################################
###################################      Estimate forecasts       ###########################################
b_names <- c("Naive", "sNaive", "SES", "MA", 
             "Croston", "optCroston","SBA", "TSB", 
             "ADIDA", "iMAPA",
             "ES_bu", "ARIMA_bu",
             "MLP_l", "RF_l")
#Get forecasts for local models
frc_total <- foreach(tsi=1:length(time_series_b), .combine='rbind', 
                     .packages=c('zoo','randomForest','RSNNS','forecast','smooth')) %dopar% benchmarks_f(time_series_b[[tsi]], 28)
#Get forecasts for global models
frc_total_g <- ML_Global(28)
frc_total$MLP_g <- frc_total_g$MLP_g
frc_total$RF_g <- frc_total_g$RF_g
frc_total_g <- NULL
#Get forecasts for top-down models
insample_top <- ts(as.numeric(colSums(sales[,6:ncol(sales)])), frequency = 7)

x_var <- calendar
x_var$snap <- x_var$snap_CA+x_var$snap_WI+x_var$snap_TX
x_var$holiday <- 0
x_var[is.na(x_var$event_type_1)==F,]$holiday <- 1
x_var <- x_var[,c("snap","holiday")]

es_f <-es(insample_top, h=28)$forecast #Exponential smoothing
esx_f <-es(insample_top, xreg=x_var, h=28)$forecast #Exponential smoothing with external variables
arima_f <- forecast(auto.arima(insample_top), h=28)$mean #ARIMA
arimax_f <- forecast(auto.arima(insample_top, xreg=as.matrix(head(x_var, length(insample_top)))),
                     h=28, xreg=as.matrix(tail(x_var, 28)))$mean #ARIMA with external variables

proportions <- unlist(lapply(c(1:length(time_series_b)), 
                             function(x) sum(tail(as.numeric(sales[x,6:ncol(sales)]),28))/sum(tail(insample_top, 28))))

frc_total$ES_td <- unlist(lapply(c(1:length(time_series_b)), function(x) es_f*proportions[x]))
frc_total$ESX <- unlist(lapply(c(1:length(time_series_b)), function(x) esx_f*proportions[x]))
frc_total$ARIMA_td <- unlist(lapply(c(1:length(time_series_b)), function(x) arima_f*proportions[x]))
frc_total$ARIMAX <- unlist(lapply(c(1:length(time_series_b)), function(x) arimax_f*proportions[x]))

#Get forecasts for combination approaches
frc_total$Com_b <- (frc_total$ES_bu+frc_total$ARIMA_bu)/2
frc_total$Com_t <- (frc_total$ES_td+frc_total$ARIMA_td)/2
frc_total$Com_tb <- (frc_total$ES_bu+frc_total$ES_td)/2
frc_total$Com_lg <- (frc_total$MLP_l+frc_total$MLP_g)/2

b_names <- c("Naive", "sNaive", "SES", "MA", 
             "Croston", "optCroston","SBA", "TSB", 
             "ADIDA", "iMAPA",
             "ES_bu", "ARIMA_bu",
             "MLP_l", "RF_l", "MLP_g", "RF_g",
             "ES_td","ESX","ARIMA_td","ARIMAX",
             "Com_b","Com_t","Com_tb","Com_lg")
frc_total <- frc_total[,c(b_names,"item_id","dept_id","cat_id","store_id","state_id","fh")]


save.image("Base Forecasts.Rdata")

#############################################################################################################


#############################################################################################################
###################################      Evaluate forecasts       ###########################################

sales_out <- read.csv("sales_test_validation.csv", stringsAsFactors = F)


#Level 12	- Unit sales of product x, aggregated for each store - 30,750
errors_total <- NULL
for (tsid in 1:nrow(sales)){
  
  insample_d <- time_series_b[[tsid]]
  insample <- insample_d$x
  outsample <- as.numeric(sales_out[tsid,6:ncol(sales_out)])
  
  Methods <- frc_total[(frc_total$item_id==insample_d$item_id)&(frc_total$store_id==insample_d$store_id),]
  
  RMSSE <- c()
  for (j in 1:length(b_names)){
    RMSSE <- c(RMSSE, sqrt(mean((Methods[,j]-outsample)^2)/mean(diff(insample)^2))) #scale MSE using first differences
  }
  errors <- data.frame(t(RMSSE)) 
  colnames(errors) <- b_names
  errors$id <- tsid 
  errors$sales <- stat_total[tsid,]$dollar_sales
  row.names(errors) <- NULL
  errors_total <- rbind(errors_total, errors)
}

WRMSSE_12 <- c()
for (mid in 1:length(b_names)){
  WRMSSE_12 <- c(WRMSSE_12, sum(errors_total[,mid]*errors_total$sales/sum(errors_total$sales)))
}
names(WRMSSE_12) <- b_names


#Level 1 - Unit sales of all products, aggregated for all stores/states	- 1
insample <- as.numeric(colSums(sales[,6:ncol(sales)]))
outsample <- as.numeric(colSums(sales_out[,6:ncol(sales_out)]))
Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh")], .(fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_1 <- c()
for (j in 1:length(b_names)){
  WRMSSE_1 <- c(WRMSSE_1, sqrt(mean((Methods[,j]-outsample)^2)/mean(diff(insample)^2)))
}


#Level 2 - Unit sales of all products, aggregated for each State - 3
insample <- sales
insample$item_id = insample$dept_id = insample$cat_id = insample$store_id <- NULL
insample <- ddply(insample, .(state_id), colwise(sum))

outsample <- sales_out
outsample$item_id = outsample$dept_id = outsample$cat_id = outsample$store_id <- NULL
outsample <- ddply(outsample, .(state_id), colwise(sum))

Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh","state_id")], .(state_id,fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_2 <- c()
for (j in 1:length(b_names)){
  temp_error <- 0
  for (i in 1:nrow(insample)){
    temp_in <- as.numeric(insample[i,2:ncol(insample)])
    sstart<-data.frame(temp_in, c(1:length(temp_in)))
    sstart <- min(sstart[sstart$temp_in>0,2])
    temp_in <- temp_in[sstart:length(temp_in)]
    
    temp_out <- as.numeric(outsample[i,2:ncol(outsample)])
    
    temp_w <- cbind(errors_total,stat_total)
    temp_w <- sum(temp_w[temp_w$state_id==insample[i,]$state_id,]$dollar_sales)/sum(temp_w$dollar_sales)
    
    temp_frc <- Methods[Methods$state_id==insample[i,]$state_id,]
    temp_frc$state_id <- NULL
    temp_frc <- as.numeric(temp_frc[,j])
    temp_error <- temp_error + sqrt(mean((temp_frc-temp_out)^2)/mean(diff(temp_in)^2))*temp_w
  }
  WRMSSE_2 <- c(WRMSSE_2, temp_error)
}


#Level 3 - Unit sales of all products, aggregated for each store - 10
insample <- sales
insample$item_id = insample$dept_id = insample$cat_id = insample$state_id <- NULL
insample <- ddply(insample, .(store_id), colwise(sum))

outsample <- sales_out
outsample$item_id = outsample$dept_id = outsample$cat_id = outsample$state_id <- NULL
outsample <- ddply(outsample, .(store_id), colwise(sum))

Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh","store_id")], .(store_id,fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_3 <- c()
for (j in 1:length(b_names)){
  temp_error <- 0
  for (i in 1:nrow(insample)){
    temp_in <- as.numeric(insample[i,2:ncol(insample)])
    sstart<-data.frame(temp_in, c(1:length(temp_in)))
    sstart <- min(sstart[sstart$temp_in>0,2])
    temp_in <- temp_in[sstart:length(temp_in)]
    
    temp_out <- as.numeric(outsample[i,2:ncol(outsample)])
    
    temp_w <- cbind(errors_total,stat_total)
    temp_w <- sum(temp_w[temp_w$store_id==insample[i,]$store_id,]$dollar_sales)/sum(temp_w$dollar_sales)
    
    temp_frc <- Methods[Methods$store_id==insample[i,]$store_id,]
    temp_frc$store_id <- NULL
    temp_frc <- as.numeric(temp_frc[,j])
    temp_error <- temp_error + sqrt(mean((temp_frc-temp_out)^2)/mean(diff(temp_in)^2))*temp_w
  }
  WRMSSE_3 <- c(WRMSSE_3, temp_error)
}

#Level 4 - Unit sales of all products, aggregated for each category - 3
insample <- sales
insample$item_id = insample$dept_id = insample$store_id = insample$state_id <- NULL
insample <- ddply(insample, .(cat_id), colwise(sum))

outsample <- sales_out
outsample$item_id = outsample$dept_id = outsample$store_id = outsample$state_id <- NULL
outsample <- ddply(outsample, .(cat_id), colwise(sum))

Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh","cat_id")], .(cat_id,fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_4 <- c()
for (j in 1:length(b_names)){
  temp_error <- 0
  for (i in 1:nrow(insample)){
    temp_in <- as.numeric(insample[i,2:ncol(insample)])
    sstart<-data.frame(temp_in, c(1:length(temp_in)))
    sstart <- min(sstart[sstart$temp_in>0,2])
    temp_in <- temp_in[sstart:length(temp_in)]
    
    temp_out <- as.numeric(outsample[i,2:ncol(outsample)])
    
    temp_w <- cbind(errors_total,stat_total)
    temp_w <- sum(temp_w[temp_w$cat_id==insample[i,]$cat_id,]$dollar_sales)/sum(temp_w$dollar_sales)
    
    temp_frc <- Methods[Methods$cat_id==insample[i,]$cat_id,]
    temp_frc$cat_id <- NULL
    temp_frc <- as.numeric(temp_frc[,j])
    temp_error <- temp_error + sqrt(mean((temp_frc-temp_out)^2)/mean(diff(temp_in)^2))*temp_w
  }
  WRMSSE_4 <- c(WRMSSE_4, temp_error)
}


#Level 5 - Unit sales of all products, aggregated for each department - 7
insample <- sales
insample$item_id = insample$cat_id = insample$store_id = insample$state_id <- NULL
insample <- ddply(insample, .(dept_id), colwise(sum))

outsample <- sales_out
outsample$item_id = outsample$cat_id = outsample$store_id = outsample$state_id <- NULL
outsample <- ddply(outsample, .(dept_id), colwise(sum))

Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh","dept_id")], .(dept_id,fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_5 <- c()
for (j in 1:length(b_names)){
  temp_error <- 0
  for (i in 1:nrow(insample)){
    temp_in <- as.numeric(insample[i,2:ncol(insample)])
    sstart<-data.frame(temp_in, c(1:length(temp_in)))
    sstart <- min(sstart[sstart$temp_in>0,2])
    temp_in <- temp_in[sstart:length(temp_in)]
    
    temp_out <- as.numeric(outsample[i,2:ncol(outsample)])
    
    temp_w <- cbind(errors_total,stat_total)
    temp_w <- sum(temp_w[temp_w$dept_id==insample[i,]$dept_id,]$dollar_sales)/sum(temp_w$dollar_sales)
    
    temp_frc <- Methods[Methods$dept_id==insample[i,]$dept_id,]
    temp_frc$dept_id <- NULL
    temp_frc <- as.numeric(temp_frc[,j])
    temp_error <- temp_error + sqrt(mean((temp_frc-temp_out)^2)/mean(diff(temp_in)^2))*temp_w
  }
  WRMSSE_5 <- c(WRMSSE_5, temp_error)
}


#Level 6 - Unit sales of all products, aggregated for each State and category - 9
insample <- sales
insample$item_id = insample$store_id = insample$dept_id <- NULL
insample <- ddply(insample, .(cat_id, state_id), colwise(sum))

outsample <- sales_out
outsample$item_id = outsample$store_id = outsample$dept_id <- NULL
outsample <- ddply(outsample, .(cat_id, state_id), colwise(sum))

Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh","cat_id","state_id")], .(cat_id,state_id,fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_6 <- c()
for (j in 1:length(b_names)){
  temp_error <- 0
  for (i in 1:nrow(insample)){
    temp_in <- as.numeric(insample[i,3:ncol(insample)])
    sstart<-data.frame(temp_in, c(1:length(temp_in)))
    sstart <- min(sstart[sstart$temp_in>0,2])
    temp_in <- temp_in[sstart:length(temp_in)]
    
    temp_out <- as.numeric(outsample[i,3:ncol(outsample)])
    
    temp_w <- cbind(errors_total,stat_total)
    temp_w <- sum(temp_w[(temp_w$cat_id==insample[i,]$cat_id)&(temp_w$state_id==insample[i,]$state_id),]$dollar_sales)/sum(temp_w$dollar_sales)
    
    temp_frc <- Methods[(Methods$cat_id==insample[i,]$cat_id)&(Methods$state_id==insample[i,]$state_id),]
    temp_frc$state_id = temp_frc$cat_id <- NULL
    temp_frc <- as.numeric(temp_frc[,j])
    temp_error <- temp_error + sqrt(mean((temp_frc-temp_out)^2)/mean(diff(temp_in)^2))*temp_w
  }
  WRMSSE_6 <- c(WRMSSE_6, temp_error)
}


#Level 7 - Unit sales of all products, aggregated for each State and department - 21
insample <- sales
insample$item_id = insample$store_id = insample$cat_id <- NULL
insample <- ddply(insample, .(dept_id, state_id), colwise(sum))

outsample <- sales_out
outsample$item_id = outsample$store_id = outsample$cat_id <- NULL
outsample <- ddply(outsample, .(dept_id, state_id), colwise(sum))

Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh","dept_id","state_id")], .(dept_id,state_id,fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_7 <- c()
for (j in 1:length(b_names)){
  temp_error <- 0
  for (i in 1:nrow(insample)){
    temp_in <- as.numeric(insample[i,3:ncol(insample)])
    sstart<-data.frame(temp_in, c(1:length(temp_in)))
    sstart <- min(sstart[sstart$temp_in>0,2])
    temp_in <- temp_in[sstart:length(temp_in)]
    
    temp_out <- as.numeric(outsample[i,3:ncol(outsample)])
    
    temp_w <- cbind(errors_total,stat_total)
    temp_w <- sum(temp_w[(temp_w$dept_id==insample[i,]$dept_id)&(temp_w$state_id==insample[i,]$state_id),]$dollar_sales)/sum(temp_w$dollar_sales)
    
    temp_frc <- Methods[(Methods$dept_id==insample[i,]$dept_id)&(Methods$state_id==insample[i,]$state_id),]
    temp_frc$state_id = temp_frc$dept_id <- NULL
    temp_frc <- as.numeric(temp_frc[,j])
    temp_error <- temp_error + sqrt(mean((temp_frc-temp_out)^2)/mean(diff(temp_in)^2))*temp_w
  }
  WRMSSE_7 <- c(WRMSSE_7, temp_error)
}


#Level 8 - Unit sales of all products, aggregated for each store and category - 30
insample <- sales
insample$item_id = insample$state_id = insample$dept_id <- NULL
insample <- ddply(insample, .(cat_id, store_id), colwise(sum))

outsample <- sales_out
outsample$item_id = outsample$state_id = outsample$dept_id <- NULL
outsample <- ddply(outsample, .(cat_id, store_id), colwise(sum))

Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh","cat_id","store_id")], .(cat_id,store_id,fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_8 <- c()
for (j in 1:length(b_names)){
  temp_error <- 0
  for (i in 1:nrow(insample)){
    temp_in <- as.numeric(insample[i,3:ncol(insample)])
    sstart<-data.frame(temp_in, c(1:length(temp_in)))
    sstart <- min(sstart[sstart$temp_in>0,2])
    temp_in <- temp_in[sstart:length(temp_in)]
    
    temp_out <- as.numeric(outsample[i,3:ncol(outsample)])
    
    temp_w <- cbind(errors_total,stat_total)
    temp_w <- sum(temp_w[(temp_w$cat_id==insample[i,]$cat_id)&(temp_w$store_id==insample[i,]$store_id),]$dollar_sales)/sum(temp_w$dollar_sales)
    
    temp_frc <- Methods[(Methods$cat_id==insample[i,]$cat_id)&(Methods$store_id==insample[i,]$store_id),]
    temp_frc$store_id = temp_frc$cat_id <- NULL
    temp_frc <- as.numeric(temp_frc[,j])
    temp_error <- temp_error + sqrt(mean((temp_frc-temp_out)^2)/mean(diff(temp_in)^2))*temp_w
  }
  WRMSSE_8 <- c(WRMSSE_8, temp_error)
}


#Level 9 - Unit sales of all products, aggregated for each store and department - 70
insample <- sales
insample$item_id = insample$state_id = insample$cat_id <- NULL
insample <- ddply(insample, .(dept_id, store_id), colwise(sum))

outsample <- sales_out
outsample$item_id = outsample$state_id = outsample$cat_id <- NULL
outsample <- ddply(outsample, .(dept_id, store_id), colwise(sum))

Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh","dept_id","store_id")], .(dept_id,store_id,fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_9 <- c()
for (j in 1:length(b_names)){
  temp_error <- 0
  for (i in 1:nrow(insample)){
    temp_in <- as.numeric(insample[i,3:ncol(insample)])
    sstart<-data.frame(temp_in, c(1:length(temp_in)))
    sstart <- min(sstart[sstart$temp_in>0,2])
    temp_in <- temp_in[sstart:length(temp_in)]
    
    temp_out <- as.numeric(outsample[i,3:ncol(outsample)])
    
    temp_w <- cbind(errors_total,stat_total)
    temp_w <- sum(temp_w[(temp_w$dept_id==insample[i,]$dept_id)&(temp_w$store_id==insample[i,]$store_id),]$dollar_sales)/sum(temp_w$dollar_sales)
    
    temp_frc <- Methods[(Methods$dept_id==insample[i,]$dept_id)&(Methods$store_id==insample[i,]$store_id),]
    temp_frc$store_id = temp_frc$dept_id <- NULL
    temp_frc <- as.numeric(temp_frc[,j])
    temp_error <- temp_error + sqrt(mean((temp_frc-temp_out)^2)/mean(diff(temp_in)^2))*temp_w
  }
  WRMSSE_9 <- c(WRMSSE_9, temp_error)
}


#10 - Unit sales of product x, aggregated for all stores/states - 3,049
insample <- sales
insample$dept_id = insample$state_id = insample$cat_id = insample$store_id<- NULL
insample <- ddply(insample, .(item_id), colwise(sum))

outsample <- sales_out
outsample$dept_id = outsample$state_id = outsample$cat_id = outsample$store_id <- NULL
outsample <- ddply(outsample, .(item_id), colwise(sum))

Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh","item_id")], .(item_id,fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_10 <- c()
for (j in 1:length(b_names)){
  temp_error <- 0
  for (i in 1:nrow(insample)){
    temp_in <- as.numeric(insample[i,2:ncol(insample)])
    sstart<-data.frame(temp_in, c(1:length(temp_in)))
    sstart <- min(sstart[sstart$temp_in>0,2])
    temp_in <- temp_in[sstart:length(temp_in)]
    
    temp_out <- as.numeric(outsample[i,2:ncol(outsample)])
    
    temp_w <- cbind(errors_total,stat_total)
    temp_w <- sum(temp_w[temp_w$item_id==insample[i,]$item_id,]$dollar_sales)/sum(temp_w$dollar_sales)
    
    temp_frc <- Methods[Methods$item_id==insample[i,]$item_id,]
    temp_frc$item_id <- NULL
    temp_frc <- as.numeric(temp_frc[,j])
    temp_error <- temp_error + sqrt(mean((temp_frc-temp_out)^2)/mean(diff(temp_in)^2))*temp_w
  }
  WRMSSE_10 <- c(WRMSSE_10, temp_error)
}


#11 - Unit sales of product x, aggregated for each State - 9,225
insample <- sales
insample$dept_id = insample$cat_id = insample$store_id<- NULL
insample <- ddply(insample, .(item_id, state_id), colwise(sum))

outsample <- sales_out
outsample$dept_id = outsample$cat_id = outsample$store_id <- NULL
outsample <- ddply(outsample, .(item_id, state_id), colwise(sum))

Methods <- ddply(frc_total[,c(colnames(frc_total)[1:length(b_names)],"fh","item_id","state_id")], .(item_id,state_id,fh), colwise(sum))
Methods$fh <- NULL

WRMSSE_11 <- c()
for (j in 1:length(b_names)){
  temp_error  <- 0
  for (i in 1:nrow(insample)){
    temp_in <- as.numeric(insample[i,3:ncol(insample)])
    sstart<-data.frame(temp_in, c(1:length(temp_in)))
    sstart <- min(sstart[sstart$temp_in>0,2])
    temp_in <- temp_in[sstart:length(temp_in)]
    
    temp_out <- as.numeric(outsample[i,3:ncol(outsample)])
    
    temp_w <- cbind(errors_total,stat_total)
    temp_w <- sum(temp_w[(temp_w$item_id==insample[i,]$item_id)&(temp_w$state_id==insample[i,]$state_id),]$dollar_sales)/sum(temp_w$dollar_sales)
    
    temp_frc <- Methods[(Methods$item_id==insample[i,]$item_id)&(Methods$state_id==insample[i,]$state_id),]
    temp_frc$item_id = temp_frc$state_id <- NULL
    temp_frc <- as.numeric(temp_frc[,j])
    temp_error <- temp_error + sqrt(mean((temp_frc-temp_out)^2)/mean(diff(temp_in)^2))*temp_w
  }
  WRMSSE_11 <- c(WRMSSE_11, temp_error)
}

WRMSSE <- rbind(WRMSSE_1, WRMSSE_2, WRMSSE_3,
                WRMSSE_4, WRMSSE_5, WRMSSE_6,
                WRMSSE_7, WRMSSE_8, WRMSSE_9,
                WRMSSE_10, WRMSSE_11, WRMSSE_12)
WRMSSE <- rbind(WRMSSE, colMeans(WRMSSE))
row.names(WRMSSE) <- c("Total", "State", "Store",
                       "Category", "Department", "State-Category",
                       "State-Department", "Store-Category", "Store-Department",
                       "Product", "Product-State", "Product-Store", "Average")

write.csv(WRMSSE, "summary.csv")
write.csv(stat_total, "stat_total.csv")
save.image("Evaluation WRMSSE.Rdata")


#############################################################################################################
# Export benchmarks' forecasts in Kaggle format
for (mid in 1:length(b_names)){
  submission <- frc_total[,c("item_id", "store_id", b_names[mid], "fh")]
  colnames(submission)[1:2] <- c("Agg_Level_1", "Agg_Level_2")
  submission$F7 = submission$F6 = submission$F5 = submission$F4 = submission$F3 = submission$F2 = submission$F1<- NA
  submission$F14 = submission$F13 = submission$F12 = submission$F11 = submission$F10 = submission$F9 = submission$F8 <- NA
  submission$F21 = submission$F20 = submission$F19 = submission$F18 = submission$F17 = submission$F16 = submission$F15 <- NA
  submission$F28 = submission$F27 = submission$F26 = submission$F25 = submission$F24 = submission$F23 = submission$F22 <- NA
  
  l1_unique <- unique(submission$Agg_Level_1)
  l2_unique <- unique(submission$Agg_Level_2)
  frc <- NULL
  for (l2 in l2_unique){
    for (l1 in l1_unique){
      temp <- submission[(submission$Agg_Level_1==l1)&(submission$Agg_Level_2==l2),]
      temp[1,5:32] <- temp[,3]
      frc <- rbind(frc, data.frame(l1, l2, temp[1,5:32]))
    }
  }
  colnames(frc)[1:2] <- c("Agg_Level_1", "Agg_Level_2")
  frc$id <- paste0(frc$Agg_Level_1,"_",frc$Agg_Level_2,"_validation")
  frc <- frc[,c("id", colnames(frc)[3:30])]
  write.csv(frc, row.names = FALSE, paste0("PF_", b_names[mid],".csv")) 
}
