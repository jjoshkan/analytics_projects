#==/* Descriptive Statistics and SLR

rm(list=ls(all=TRUE)) #clears memory
options(digits=12) #keeps 12 digits in memory, otherwise summary statistics may be off due to rounding

# read the csv file
stockdata <- read.csv("WFS - 2015 to 2020 Stock Data - Adj. Close.csv", header=TRUE) 
head(stockdata)

# Remove the first row
newstockdata <- stockdata[-1,]
head(newstockdata)
colnames(newstockdata)

# Compute descriptive statistics 
round(mean(newstockdata[,"AAPL.Return.percent"]), 1)
round(summary(newstockdata[,"AAPL.Return.percent"],1))
round(sqrt(var(newstockdata[,"AAPL.Return.percent"])), 1)
round(cor(newstockdata[,"AAPL.Return.percent"], newstockdata[,"SP.500.Return.percent"]), 1)
round(sqrt(var(newstockdata[,"AAPL.Return.percent"]))/mean(newstockdata[,"AAPL.Return.percent"]), 1)

round(mean(newstockdata[,"INTC.Return.percent"]), 1)
round(summary(newstockdata[,"INTC.Return.percent"],1))
round(sqrt(var(newstockdata[,"INTC.Return.percent"])), 1)
round(cor(newstockdata[,"INTC.Return.percent"], newstockdata[,"SP.500.Return.percent"]), 1)
round(sqrt(var(newstockdata[,"INTC.Return.percent"]))/mean(newstockdata[,"INTC.Return.percent"]), 1)

round(mean(newstockdata[,"KR.Return.percent"]), 1)
round(summary(newstockdata[,"KR.Return.percent"],1))
round(sqrt(var(newstockdata[,"KR.Return.percent"])), 1)
round(cor(newstockdata[,"KR.Return.percent"], newstockdata[,"SP.500.Return.percent"]), 1)
round(sqrt(var(newstockdata[,"KR.Return.percent"]))/mean(newstockdata[,"KR.Return.percent"]), 1)

round(mean(newstockdata[,"SP.500.Return.percent"]), 1)
round(summary(newstockdata[,"SP.500.Return.percent"],1))
round(sqrt(var(newstockdata[,"SP.500.Return.percent"])), 1)
round(cor(newstockdata[,"SP.500.Return.percent"], newstockdata[,"SP.500.Return.percent"]), 1)
round(sqrt(var(newstockdata[,"SP.500.Return.percent"]))/mean(newstockdata[,"SP.500.Return.percent"]), 1)

# confirm that variable called Return is numeric
str(newstockdata$SP.500.Return)
str(newstockdata$AAPL.Return)
str(newstockdata$INTC.Return)
str(newstockdata$KR.Return)

newstockdata <- transform(newstockdata, SP.500.Return.plus.1 = SP.500.Return + 1)
newstockdata <- transform(newstockdata, AAPL.Return.plus.1 = AAPL.Return + 1)
newstockdata <- transform(newstockdata, INTC.Return.plus.1 = INTC.Return + 1)
newstockdata <- transform(newstockdata, KR.Return.plus.1 = KR.Return + 1)

geomeanreturn_sp500 <- exp(mean(log(newstockdata$SP.500.Return.plus.1))) - 1
geomeanreturn_sp500 <- round(geomeanreturn_sp500 * 100,1)
geomeanreturn_aapl <- exp(mean(log(newstockdata$AAPL.Return.plus.1))) - 1
geomeanreturn_aapl <- round(geomeanreturn_aapl * 100,1)
geomeanreturn_intc <- exp(mean(log(newstockdata$INTC.Return.plus.1))) - 1
geomeanreturn_intc <- round(geomeanreturn_intc * 100,1)
geomeanreturn_kr <- exp(mean(log(newstockdata$KR.Return.plus.1))) - 1
geomeanreturn_kr <- round(geomeanreturn_kr * 100,1)

beta_aapl <- cov(newstockdata["AAPL.Return.percent"],newstockdata["SP.500.Return.percent"])/var(newstockdata["SP.500.Return.percent"])
beta_aapl <- round(beta_aapl,1)
beta_aapl
beta_intc <- cov(newstockdata["INTC.Return.percent"],newstockdata["SP.500.Return.percent"])/var(newstockdata["SP.500.Return.percent"])
beta_intc <- round(beta_intc,1)
beta_intc
beta_kr <- cov(newstockdata["KR.Return.percent"],newstockdata["SP.500.Return.percent"])/var(newstockdata["SP.500.Return.percent"])
beta_kr <- round(beta_kr,1)
beta_kr
beta_sp500 <- cov(newstockdata["SP.500.Return.percent"],newstockdata["SP.500.Return.percent"])/var(newstockdata["SP.500.Return.percent"])
beta_sp500 <- round(beta_sp500,1)
beta_sp500

# Extract variables to be used in the analyses
sp500 <- newstockdata[,"SP.500.Return.percent"]
appl <- newstockdata[,"AAPL.Return.percent"]
intc <- newstockdata[,"INTC.Return.percent"]
kr <- newstockdata[,"KR.Return.percent"]
sp500 <- newstockdata[,"SP.500.Return.plus.1"]
appl <- newstockdata[,"AAPL.Return.plus.1"]
intc <- newstockdata[,"INTC.Return.plus.1"]
kr <- newstockdata[,"KR.Return.plus.1"]
# sp500 <- newstockdata[,"SP.500.Return"]
# appl <- newstockdata[,"AAPL.Return"]
# intc <- newstockdata[,"INTC.Return"]
# kr <- newstockdata[,"KR.Return"]

# Regress sp500 on appl
mod.1 <- lm(appl ~ sp500)
summary(mod.1) # Present Parameter Estimates, Coefficient of Determination, etc.
# Regress sp500 on intc
mod.2 <- lm(intc ~ sp500)
summary(mod.2) # Present Parameter Estimates, Coefficient of Determination, etc.
# Regress sp500 on kr
mod.3 <- lm(kr ~ sp500)
summary(mod.3) # Present Parameter Estimates, Coefficient of Determination, etc.

# Plot data and regression line
plot(sp500, appl)
abline(mod.1, col="red")

plot(sp500, intc)
abline(mod.2, col="blue")

plot(sp500, kr)
abline(mod.3, col="orange")

# Test APPL Beta = 1
beta_appl <- summary(mod.1)$coefficients[2, 1]
se_appl <- summary(mod.1)$coefficients[2, 2]
t_appl <- (beta_appl - 1) / se_appl
p_value_appl <- 2 * pt(-abs(t_appl), df = nrow(newstockdata) - 2)
p_value_appl

# Test INTC Beta = 1
beta_intc <- summary(mod.2)$coefficients[2, 1]
se_intc <- summary(mod.2)$coefficients[2, 2]
t_intc <- (beta_intc - 1) / se_intc
p_value_intc <- 2 * pt(-abs(t_intc), df = nrow(newstockdata) - 2)
p_value_intc

# Test KR Beta = 1
beta_kr <- summary(mod.3)$coefficients[2, 1]
se_kr <- summary(mod.3)$coefficients[2, 2]
t_kr <- (beta_kr - 1) / se_kr
p_value_kr <- 2 * pt(-abs(t_kr), df = nrow(newstockdata) - 2)
p_value_kr

