faces <- read.csv("faces.csv", header = F)

# Function that normalizes a vector x (i.e. |x|=1 ) #
normalize <- function(x) {
  norm = sqrt(sum(x^2))
  x/norm
}

# Display first face
first_face <- matrix(unlist(faces[1,]),64,64)
image(first_face, col=gray((0:255)/255))

# Display a random face
random_face <- matrix(unlist(faces[sample(1:400,1),]),64,64)
image(random_face, col=gray((0:255)/255))

# Compute and display mean face
mean_face <- matrix(unlist(colMeans(faces)),64,64)
image(x=1:64,y=1:64 , mean_face, col=gray((0:255)/255))

# Centralize the faces by subtracting the mean face
mean <- matrix(rep(as.vector(mean_face),400),nrow(faces),ncol(faces),byrow = T)
central_faces <- faces - mean 

# Calculate PCs #
central_faces <- as.matrix(central_faces)
temp <- central_faces %*% t(central_faces)
temp_eig <- eigen(temp)
PC_scores <- temp_eig$values
PC <- t(central_faces)%*%(temp_eig$vectors)

for (i in 1:ncol(PC)) {
  PC[,i] <- normalize(PC[,i])
}

# Display first eigenfaces #
image(matrix(PC[,50],64,64), col=gray((0:255)/255))

# Reconstruct first face using first k PCs #
k = 200; # k can be any value between 1 and 400
w <- t(PC[,1:k])%*%(t(as.matrix(faces[1,]))-as.matrix(colMeans(faces)))

constr_face <- vector("numeric", length = ncol(faces))
for (i in 1:k) {
  constr_face <- constr_face + w[i]*PC[,i]
}

constr_face <- t(constr_face) + colMeans(faces)
image(matrix(constr_face,64,64), col=gray((0:255)/255))

# Plot proportion of variance from first 10 PCs #
var <- PC_scores/sum(PC_scores)
total_var <- vector("numeric", length(var))
total_var[1] <- var[1]

for (i in 2:length(var)) {
  total_var[i] <- var[i] + total_var[i-1] 
}

plot(total_var[1:399],pch = '.');
lines(total_var[1:399])






