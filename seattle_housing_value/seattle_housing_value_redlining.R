#==/* Descriptive Statistics and SLR

rm(list=ls(all=TRUE)) #clears memory
options(digits=12) #keeps 12 digits in memory, otherwise summary statistics may be off due to rounding

####################################################################################################################################################################

# read the csv file
housedata <- read.csv("Redlining Data.csv", header=TRUE) 
dim(housedata)

housedata <- housedata[, c("redlined", "lotsizesquarefeet",	"buildingareasqft",	"totalbedrooms",	"salespriceamount",	"TotalCalculatedBathCount",
                           "Renovated",	"Single_family",	"Townhome",	"BC_Excellent",	"BC_Good",	"BC_Average",	"BC_Fair",	"BC_NA", "Home_Age")]
dim(housedata)

housedata <- na.omit(housedata)
housedata <- housedata[housedata$BC_NA == 0,]
dim(housedata)

housedata2 <- housedata[
                        housedata$salespriceamount <= 10000000 &
                        housedata$totalbedrooms > 0 &
                        housedata$TotalCalculatedBathCount > 0 &
                        housedata$Home_Age >= 0 &
                        housedata$lotsizesquarefeet >= 500 &
                        housedata$lotsizesquarefeet <= 50000,
                      ]
dim(housedata2)

library(stargazer)
stargazer(housedata2, type = "text", title = "Descriptive Statistics", summary = TRUE)

####################################################################################################################################################################

# Step 1: Fit the regression model
mod.3 <- lm(salespriceamount ~ redlined + lotsizesquarefeet + buildingareasqft +
              totalbedrooms + TotalCalculatedBathCount + Renovated +
              Single_family + Townhome + BC_Excellent + BC_Good +
              BC_Average + BC_Fair + Home_Age + I(Home_Age^2), data = housedata2)

# Step 2: Conduct the Breusch-Pagan test
bptest(mod.3)

# Step 3: Compute robust standard errors
robust_vcov3 <- vcovHC(mod.3, type = "HC")  # Heteroskedasticity-consistent covariance matrix
robust_se3 <- sqrt(diag(robust_vcov3))       # Robust standard errors
print(robust_se3)

# Step 4: Correct t-statistics and p-values
coeftest(mod.3, vcov = robust_vcov3)

################################################################################

# Step 1: Fit the regression model
mod.4 <- lm(log(salespriceamount) ~ redlined + lotsizesquarefeet + buildingareasqft +
              totalbedrooms + TotalCalculatedBathCount + Renovated +
              Single_family + Townhome + BC_Excellent + BC_Good +
              BC_Average + BC_Fair + Home_Age + I(Home_Age^2), data = housedata2)

# Step 2: Conduct the Breusch-Pagan test
bptest(mod.4)

# Step 3: Compute standard error
standard_se4 <- sqrt(diag(vcov(mod.4)))

library(stargazer)
stargazer(mod.3, mod.4,
          se = list(robust_se3,standard_se4),
          title = "Regression Results: Linear and Log-Linear Models of Housing Prices",
          align = TRUE,
          dep.var.labels = c("Sales Price", "log(Sales Price)"),
          covariate.labels = c("Redlined Area", 
                               "Lot Size", 
                               "Building Area", 
                               "Total Bedrooms", 
                               "Total Bathrooms", 
                               "Renovation Status", 
                               "Single-Family Home", 
                               "Townhome", 
                               "Building Condition: Excellent", 
                               "Building Condition: Good", 
                               "Building Condition: Average", 
                               "Building Condition: Fair", 
                               "Home Age", 
                               "Home Age Squared"),
          omit.stat = c("LL", "ser", "f"),
          digits = 3,
          no.space = TRUE,
          type = "text",
          star.cutoffs = c(0.05, 0.01, 0.001),
          add.lines = list(
            c("Note", "Robust SE used for Linear Model", "SE used for Log-Linear Model")
          ))

####################################################################################################################################################################

# Technical Appendix

# Linear Model
robust_vcov3
coeftest(mod.3, vcov = robust_vcov3)
write.csv(robust_vcov3, "robust_vcov3.csv", row.names = TRUE)


# Log-Linear Model
robust_vcov4 <- vcovHC(mod.4, type = "HC")
coeftest(mod.4, vcov = robust_vcov4)