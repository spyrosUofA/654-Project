# Title     : T
# Objective : Plot
# Created by: spyros
# Created on: 2021-04-14
getwd()

r = 5


non_private = 1/r*read.table("./save/NoAttacks/PrivateFL_mnist_cnn_seed0.txt", quote="\"")

#### LDP ####
# 0.05 noise scale
ldp_c10.01_s0.05 = 1/r*read.table("./save/NoAttacks/LDP_iid0_mnist_cnn_norm10.01_scale0.05_seed0.txt", quote="\"")
ldp_c5.01_s0.05  = 1/r*read.table("./save/NoAttacks/LDP_iid0_mnist_cnn_norm5.01_scale0.05_seed0.txt", quote="\"")

ldp_c10.01_s0.1 = 1/r*read.table("./save/NoAttacks/LDP_iid0_mnist_cnn_norm10.01_scale0.1_seed0.txt", quote="\"")
ldp_c5.01_s0.1  = 1/r*read.table("./save/NoAttacks/LDP_iid0_mnist_cnn_norm5.01_scale0.1_seed0.txt", quote="\"")

ldp_c10.01_s0.15 = 1/r*read.table("./save/NoAttacks/LDP_iid0_mnist_cnn_norm10.01_scale0.005_seed0.txt", quote="\"")
ldp_c5.01_s0.15  = 1/r*read.table("./save/NoAttacks/LDP_iid0_mnist_cnn_norm5.01_scale0.15_seed0.txt", quote="\"")


for (seed in 1:(r-1)){
      non_private = non_private + 1/r*read.table(paste0("./save/NoAttacks/PrivateFL_mnist_cnn_seed", seed, ".txt"), quote="\"")

      ldp_c10.01_s0.05 = ldp_c10.01_s0.05 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid0_mnist_cnn_norm10.01_scale0.05_seed", seed, ".txt"), quote="\"")
      ldp_c5.01_s0.05 = ldp_c5.01_s0.05 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid0_mnist_cnn_norm5.01_scale0.05_seed", seed, ".txt"), quote="\"")

      ldp_c10.01_s0.1 = ldp_c10.01_s0.1 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid0_mnist_cnn_norm10.01_scale0.1_seed", seed, ".txt"), quote="\"")
      ldp_c5.01_s0.1 = ldp_c5.01_s0.1 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid0_mnist_cnn_norm5.01_scale0.1_seed", seed, ".txt"), quote="\"")

      ldp_c10.01_s0.15 = ldp_c10.01_s0.15 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid0_mnist_cnn_norm10.01_scale0.005_seed", seed, ".txt"), quote="\"")
      ldp_c5.01_s0.15 = ldp_c5.01_s0.15 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid0_mnist_cnn_norm5.01_scale0.15_seed", seed, ".txt"), quote="\"")
}




# LDP Plots
jpeg("./plotting/NoAttack_ldp_NON_iid.jpg")
plot(0:20, non_private$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
# 0.05 noise
lines(0:20, ldp_c5.01_s0.05$V1, col="green", lty=2)
lines(0:20, ldp_c10.01_s0.05$V1, col="green", lty=1)
# 0.10 noise
lines(0:20, ldp_c5.01_s0.1$V1, col="red", lty=2)
lines(0:20, ldp_c10.01_s0.1$V1, col="red", lty=1)
# 0.15
lines(0:20, ldp_c5.01_s0.15$V1, col="blue", lty=2)
lines(0:20, ldp_c10.01_s0.15$V1, col="blue", lty=1)
title(main="Local Differential Privacy")
legend("bottomright", c("Non-Private",
                        expression(paste(sigma, "=0.05, C=5")), expression(paste(sigma, "=0.05, C=10")),
                        expression(paste(sigma, "=0.10, C=5")), expression(paste(sigma, "=0.10, C=10")),
                        expression(paste(sigma, "=0.15, C=5")), expression(paste(sigma, "=0.15, C=10"))),
        col = c("black", "green", "green","red", "red", "blue", "blue"), lty=c(1,1,2,1,2))
dev.off()

