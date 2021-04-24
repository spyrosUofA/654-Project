# Title     : T
# Objective : Plot
# Created by: spyros
# Created on: 2021-04-14
getwd()

r = 5

#### Baseline ####
non_private = read.table("./save/NoAttacks/NonPrivate_iid1_mnist_cnn_seed0.txt", quote="\"")
non_private = non_private/2 + read.table("./save/NoAttacks/NonPrivate_iid1_mnist_cnn_seed1.txt", quote="\"")/2

#### CDP ####
# 0.15 noise scale
gdp_c3.2_s0.15 = 1/r*read.table("./save/NoAttacks/GDP_iid1_mnist_cnn_norm3.2_scale0.15_seed0.txt", quote="\"")
gdp_c1.6_s0.15 = 1/r*read.table("./save/NoAttacks/GDP_iid1_mnist_cnn_norm1.6_scale0.15_seed0.txt", quote="\"")
# 0.30 noise scale
gdp_c3.2_s0.3 = 1/r*read.table("./save/NoAttacks/GDP_iid1_mnist_cnn_norm3.2_scale0.3_seed0.txt", quote="\"")
gdp_c1.6_s0.3 = 1/r*read.table("./save/NoAttacks/GDP_iid1_mnist_cnn_norm1.6_scale0.3_seed0.txt", quote="\"")

for (seed in 1:(r-1)){
      gdp_c3.2_s0.15 = gdp_c3.2_s0.15 + 1/r*read.table(paste0("./save/NoAttacks/GDP_iid1_mnist_cnn_norm3.2_scale0.15_seed", seed, ".txt"), quote="\"")
      gdp_c1.6_s0.15 = gdp_c1.6_s0.15 + 1/r*read.table(paste0("./save/NoAttacks/GDP_iid1_mnist_cnn_norm1.6_scale0.15_seed", seed, ".txt"), quote="\"")

      gdp_c3.2_s0.3 = gdp_c3.2_s0.3 + 1/r*read.table(paste0("./save/NoAttacks/GDP_iid1_mnist_cnn_norm3.2_scale0.3_seed", seed, ".txt"), quote="\"")
      gdp_c1.6_s0.3 = gdp_c1.6_s0.3 + 1/r*read.table(paste0("./save/NoAttacks/GDP_iid1_mnist_cnn_norm1.6_scale0.3_seed", seed, ".txt"), quote="\"")
}

# CDP Plots
jpeg("./plotting/NoAttack_cdp_iid.jpg")
plot(0:20, non_private$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
# 0.15 noise
lines(0:20, gdp_c3.2_s0.15$V1, col="blue", lty=1)
lines(0:20, gdp_c1.6_s0.15$V1, col="blue", lty=2)
# 0.30 noise
lines(0:20, gdp_c3.2_s0.3$V1, col="red", lty=1)
lines(0:20, gdp_c1.6_s0.3$V1, col="red", lty=2)
title(main="Central Differential Privacy")
legend("bottomright", c("Non-Private",
                        expression(paste(sigma, "=0.15, C=3.2")), expression(paste(sigma, "=0.15, C=1.6")),
                        expression(paste(sigma, "=0.30, C=3.2")), expression(paste(sigma, "=0.30, C=1.6"))),
        col = c("black", "blue", "blue", "red", "red"), lty=c(1,1,2,1,2,1,2))
dev.off()




#### LDP ####
# 0.05 noise scale
ldp_c10.0_s0.15 = 1/r*read.table("./save/NoAttacks/LDP_iid1_mnist_cnn_norm10.0_scale0.15_seed0.txt", quote="\"")
ldp_c5.0_s0.15 = 1/r*read.table("./save/NoAttacks/LDP_iid1_mnist_cnn_norm5.0_scale0.15_seed0.txt", quote="\"")

ldp_c15.01_s0.1= 1/r*read.table("./save/NoAttacks/LDP_iid1_mnist_cnn_norm15.01_scale0.1_seed0.txt", quote="\"")
ldp_c10.01_s0.1= 1/r*read.table("./save/NoAttacks/LDP_iid1_mnist_cnn_norm10.01_scale0.1_seed0.txt", quote="\"")
ldp_c5.01_s0.1 = 1/r*read.table("./save/NoAttacks/LDP_iid1_mnist_cnn_norm5.01_scale0.1_seed0.txt", quote="\"")

ldp_c10.01_s0.15 = 1/r*read.table("./save/NoAttacks/LDP_iid1_mnist_cnn_norm10.01_scale0.15_seed0.txt", quote="\"")
ldp_c5.01_s0.15 = 1/r*read.table("./save/NoAttacks/LDP_iid1_mnist_cnn_norm5.01_scale0.15_seed0.txt", quote="\"")

ldp_c10.01_s0.3 = 1/r*read.table("./save/NoAttacks/LDP_iid1_mnist_cnn_norm10.01_scale0.3_seed0.txt", quote="\"")
ldp_c5.01_s0.3 = 1/r*read.table("./save/NoAttacks/LDP_iid1_mnist_cnn_norm5.01_scale0.3_seed0.txt", quote="\"")


for (seed in 1:(r-1)){
      ldp_c15.01_s0.1 = ldp_c15.01_s0.1 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid1_mnist_cnn_norm15.01_scale0.1_seed", seed, ".txt"), quote="\"")
      ldp_c10.01_s0.1 = ldp_c10.01_s0.1 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid1_mnist_cnn_norm10.01_scale0.1_seed", seed, ".txt"), quote="\"")
      ldp_c5.01_s0.1 = ldp_c5.01_s0.1 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid1_mnist_cnn_norm5.01_scale0.1_seed", seed, ".txt"), quote="\"")

      ldp_c10.0_s0.15 = ldp_c10.0_s0.15 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid1_mnist_cnn_norm10.0_scale0.15_seed", seed, ".txt"), quote="\"")
      ldp_c5.0_s0.15 = ldp_c5.0_s0.15 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid1_mnist_cnn_norm5.0_scale0.15_seed", seed, ".txt"), quote="\"")

      ldp_c10.01_s0.15 = ldp_c10.01_s0.15 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid1_mnist_cnn_norm10.01_scale0.15_seed", seed, ".txt"), quote="\"")
      ldp_c5.01_s0.15 = ldp_c5.01_s0.15 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid1_mnist_cnn_norm5.01_scale0.15_seed", seed, ".txt"), quote="\"")

      ldp_c10.01_s0.3 = ldp_c10.01_s0.3 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid1_mnist_cnn_norm10.01_scale0.3_seed", seed, ".txt"), quote="\"")
      ldp_c5.01_s0.3 = ldp_c5.01_s0.3 + 1/r*read.table(paste0("./save/NoAttacks/LDP_iid1_mnist_cnn_norm5.01_scale0.3_seed", seed, ".txt"), quote="\"")

}




# LDP Plots
jpeg("./plotting/NoAttack_ldp_iid.jpg")
plot(0:20, non_private$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
# 0.15 noise
#lines(0:20, ldp_c5.0_s0.15$V1, col="green", lty=1)
#lines(0:20, ldp_c10.0_s0.15$V1, col="green", lty=2)
lines(0:20, ldp_c5.01_s0.15$V1, col="blue", lty=1)
lines(0:20, ldp_c10.01_s0.15$V1, col="blue", lty=2)
# 0.30 noise
lines(0:20, ldp_c5.01_s0.3$V1, col="red", lty=1)
lines(0:20, ldp_c10.01_s0.3$V1, col="red", lty=2)
title(main="Local Differential Privacy")
legend("bottomright", c("Non-Private",
                        expression(paste(sigma, "=0.15, C=5")), expression(paste(sigma, "=0.15, C=10")),
                        expression(paste(sigma, "=0.30, C=5")), expression(paste(sigma, "=0.30, C=10"))),
        col = c("black", "blue", "blue", "red", "red"), lty=c(1,1,2,1,2))
dev.off()

