################################################
##### Machine Learning Project
##### By Zak Kaplan, Sai Lone, Ayesha Kazi
###############################################

rm(list=ls())
setwd("/Users/Zak/Documents/R")


data <- read.delim("MLProjectData.csv", sep = ",", header = TRUE)


trimmed <- data.frame(data$Location.Name,data$Major.N,data$Oth.N,data$NoCrim.N,data$Prop.N,data$Vio.N)

colnames(data)

head(trimmed)

a <- c()
for (i in 1:dim(trimmed)[1]){
	for (j in 1:dim(trimmed)[2]){
		if (toString(trimmed[i,j]) == 'N/A'){
			if (i %in% a == FALSE){
				a<- 	c(a,i)
}
}
}
}


best <- trimmed[-a,]

b <- c()
for (i in 1:dim(best)[1]){
	for (j in 2:dim(best)[2]){
		if (as.numeric(best[i,j]) <0){
			if (i %in% b == FALSE){
				b <- 	c(b,i)
}
}
}
}

b <- c()
for (i in 1:dim(best)[1]){
	for (j in 2:dim(best)[2]){
		if (toString(best[i,j]) <0){
			if (i %in% b == FALSE){
				b <- 	c(b,i)
}			
}
}
}

best <- best[-b,]

library(caret)

library("multtest")
library("fpc")
library("cluster")
library("bootcluster")
library("fossil")

colnames(best)

best$data.Major.N <- as.numeric(best$data.Major.N)
best$data.Oth.N <- as.numeric(best$data.Oth.N)
best$data.NoCrim.N <- as.numeric(best$data.NoCrim.N)
best$data.Prop.N <- as.numeric(best$data.Prop.N)
best$data.Vio.N <- as.numeric(best$data.Vio.N)

###################### K-Means


gap_kmeans <- clusGap(scale(best[,-1]), kmeans, K.max = 50, nstart = 20, B = 10)

plot(gap_kmeans, main = "Gap Statistic: kmeans")


km <- kmeans(best[,-1], centers =4, nstart = 10)


par(mfrow=c(1,1))

plot(best[ ,c("data.Vio.N", "data.NoCrim.N")], col = alpha(km$cluster,.05), pch = 1, main = "Example: k-means")
points(km$centers[, c("data.Vio.N", "data.Prop.N")], col = 1:4, pch = 8, cex = 2)

rand.index(km$cluster, as.numeric(data[,8]))
adj.rand.index(km$cluster, as.numeric(data[,8]))

############################## SOM


library(kohonen)

best.scaled <- scale(best[,-1])

som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")
best.som <- som(best.scaled, grid = som_grid, rlen = 3000)


codes <- best.som$codes[[1]]


plot(best.som, main = "Data")

par(mar = c(2, 4, 2, 2)) # Set the margin on all sides to 2

par(mfrow=c(2,2))

plot(best.som, type = "changes", main = "Best Data")

plot(best.som, type = "count")

plot(best.som, type = "mapping")

coolBlueHotRed <- function(n, alpha = 1){rainbow(n, end=4/6, alpha = alpha)[n:1]}

plot(best.som, type = "dist.neighbours", palette.name = coolBlueHotRed)


par(mfrow=c(1,1))

d <- dist(codes)
hc <- hclust(d)

plot(hc)

som_cluster <- cutree(hc, k = 4)

som_cluster


plot(nci.som, type = "mapping", col = "black", cex = .5, bgcol = rainbow(4)[som_cluster])
add.cluster.boundaries(nci.som, som_cluster)


#################################### PCA


# principal components
pc_ex <- prcomp(best[,-1], center = TRUE, scale = TRUE)


par(mfrow=c(1,1))

plot(pc_ex, main = "Best")

summary(pc_ex)

par(mar = c(4, 4, 4, 4)) # Set the margin on all sides to 2


biplot(pc_ex, main = 'Best')


############################## AVG analysis


trimmed <- data.frame(data$Location.Name,data$AvgOfMajor.N,data$AvgOfOth.N,data$AvgOfNoCrim.N,data$AvgOfProp.N,data$AvgOfVio.N)

head(trimmed)

a <- c()
for (i in 1:dim(trimmed)[1]){
	for (j in 1:dim(trimmed)[2]){
		if (toString(trimmed[i,j]) == 'N/A'){
			if (i %in% a == FALSE){
				a<- 	c(a,i)
}
}
}
}


best <- trimmed[-a,]

b <- c()
for (i in 1:dim(best)[1]){
	for (j in 2:dim(best)[2]){
		if (as.numeric(best[i,j]) <0){
			if (i %in% b == FALSE){
				b <- 	c(b,i)
}
}
}
}

b <- c()
for (i in 1:dim(best)[1]){
	for (j in 2:dim(best)[2]){
		if (toString(best[i,j]) <0){
			if (i %in% b == FALSE){
				b <- 	c(b,i)
}			
}
}
}

best <- best[-b,]


best$data.AvgOfMajor.N <- as.numeric(best$data.AvgOfMajor.N)
best$data.AvgOfOth.N <- as.numeric(best$data.AvgOfOth.N)
best$data.AvgOfNoCrim.N <- as.numeric(best$data.AvgOfNoCrim.N)
best$data.AvgOfProp.N <- as.numeric(best$data.AvgOfProp.N)
best$data.AvgOfVio.N <- as.numeric(best$data.AvgOfVio.N)

############################### K-Means

gap_kmeans <- clusGap(scale(best[,-1]), kmeans, K.max = 10, nstart = 20, B = 100)

plot(gap_kmeans, main = "Gap Statistic: kmeans")


km <- kmeans(best[,-1], centers =4, nstart = 10)


par(mfrow=c(1,1))

plot(best[ ,c("data.AvgOfVio.N", "data.AvgOfNoCrim.N")], col = alpha(km$cluster,.05), pch = 1, main = "Example: k-means")
points(km$centers[, c("data.AvgOfVio.N", "data.AvgOfProp.N")], col = 1:4, pch = 8, cex = 2)


######################################## SOM


library(kohonen)

best.scaled <- scale(best[,-1])

som_grid <- somgrid(xdim = 6, ydim = 6, topo = "hexagonal")
best.som <- som(best.scaled, grid = som_grid, rlen = 3000)


codes <- best.som$codes[[1]]


plot(best.som, main = "Data")

par(mar = c(2, 4, 2, 2)) # Set the margin on all sides to 2

par(mfrow=c(2,2))

plot(best.som, type = "changes", main = "Best Data")

plot(best.som, type = "count")

plot(best.som, type = "mapping")

coolBlueHotRed <- function(n, alpha = 1){rainbow(n, end=4/6, alpha = alpha)[n:1]}

plot(best.som, type = "dist.neighbours", palette.name = coolBlueHotRed)


par(mfrow=c(1,1))

d <- dist(codes)
hc <- hclust(d)

plot(hc)

som_cluster <- cutree(hc, k = 4)

som_cluster


plot(nci.som, type = "mapping", col = "black", cex = .5, bgcol = rainbow(4)[som_cluster])
add.cluster.boundaries(nci.som, som_cluster)

########################################### PCA


# principal components
pc_ex <- prcomp(best[,-1], center = TRUE, scale = TRUE)


par(mfrow=c(1,1))

plot(pc_ex, main = "Variance Distribution")

summary(pc_ex)

par(mar = c(4, 4, 4, 4)) # Set the margin on all sides to 2


biplot(pc_ex, main = 'PC1 / PC2 Graph')

plot(pc_ex$x[,1],pc_ex$x[,2], xlab="PC1", 
ylab = "PC2", main = "PC1 / PC2 - plot")


index = c()
for (i in 1:length(pc_ex$x[,2])){

	if (pc_ex$x[,2][i] >= 1) {
		index <- c(index,i)


}

}


class <- c(1:length(pc_ex$x[,2]))

for (i in 1:length(pc_ex$x[,2])){

	if (i %in% index) {
		class[i] <- 1


}
	else {

		class[i] = -1
}

}


######################### SVM

library(e1071)

PC_Test <- data.frame(pc_ex$x[,1],pc_ex$x[,2],class)

colnames(PC_Test) <- c('PC1','PC2','Class')

PC_Test$Class <- as.factor(PC_Test$Class)


svmfit = svm(Class ~ ., data = PC_Test, kernel = "linear", cost = 10, scale = FALSE)

plot(svmfit, PC_Test)

make.grid = function(x, n = 75) {
  grange = apply(x, 2, range)
  x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
  x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
  expand.grid(X1 = x1, X2 = x2)
}

xgrid = make.grid(PC_Test[,1:2])
xgrid[1:10,]

colnames(xgrid) <- c('PC1','PC2')

palette(c("black","green"))
ygrid = predict(svmfit, xgrid)
plot(xgrid, col = c("red","blue")[as.numeric(ygrid)], pch = 20, cex = .2, main = "SVM Analysis")
points(PC_Test[,1:2], col = PC_Test$Class, pch = 19)
points(PC_Test[,1:2][svmfit$index,], pch = 5, cex = 2)

palette("default")


######################### KNN


library("ggplot2")
library(ggpubr)


minX1 <- min(PC_Test$PC1)
minX2 <- min(PC_Test$PC2)
maxX1 <- max(PC_Test$PC1)
maxX2 <- max(PC_Test$PC2)


# ranges
X1.range <- seq(from = minX1, to = maxX1, length.out = 100)
X2.range <- seq(from = minX2, to = maxX2, length.out = 100)


# Create the test set
test <- data.frame(X1 = rep(X1.range, 100), X2 = rep(X2.range, each = 100))
g2 <- ggplot(test, aes(X1,X2)) + geom_point(size = 0.5)
plot(g2)

require(class)
knnplot <- function(train, test, k){
	 KNN <- knn(train[, c('PC1', 'PC2')], test, train$Class, k)
	 test$predict <- KNN
	 
	 # change factor to numeric
	 test$z <- c(0, 1)[sapply(test$predict, as.numeric)]
	
	 title = paste("k=", as.character(k), sep ="")
	 
	 g <- ggplot(data = test, aes(PC1,PC2)) + geom_point(aes(colour = predict), size = 0.5) + geom_contour(aes(z=z), colour = 'black', size = 0.1) + theme(legend.position = "none") + labs(title = title)
	
	#add the training points in
	g <- g + geom_point(data = train, aes(PC1,PC2,colour = as.factor(Class), shape = 'x'))

	return(g)

}

colnames(test) <- c('PC1','PC2')

filer <- paste("k", c(1:10), ".png", sep="")
for (i in 1:10){
	p <- knnplot(PC_Test, test, i)
	ggsave(filename = filer[i], plot = p, height = 5, width = 5)
}

p1 <- knnplot(PC_Test, test, 1)
p2 <- knnplot(PC_Test, test, 2)
p3 <- knnplot(PC_Test, test, 3)
p4 <- knnplot(PC_Test, test, 4)
p5 <- knnplot(PC_Test, test, 5)
p6 <- knnplot(PC_Test, test, 6)
p7 <- knnplot(PC_Test, test, 7)
p8 <- knnplot(PC_Test, test, 8)
p9 <- knnplot(PC_Test, test, 9)

ggarrange(p1,p2,p3,p4,p5,p6,p7,p8,p9)

ggarrange(p3,p4,p5,p6)


############################################ Classification Tree

library("rpart")

best$Class <- class

model.control <- rpart.control(minsplit = 5, xval = 10, cp = 0)
fit.data <- rpart(Class~., data = best, method = "class", control = model.control)

plot(fit.data, uniform = T, compress = T)
text(fit.data, cex = 1)







