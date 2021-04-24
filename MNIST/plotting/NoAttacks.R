# Title     : T
# Objective : Plot
# Created by: spyros
# Created on: 2021-04-14
getwd()

#### Baseline ####
non_private = 0.2*read.table("./save/NoAttacks/PrivateFL_mnist_cnn_seed0.txt", quote="\"")

#### LDP ####
# 0.05 noise scale
ldp_c3.2_s0.05 = 0.2*read.table("./save/NoAttacks/LDP_FL_mnist_cnn_norm3.2_scale0.05_seed0.txt", quote="\"")
ldp_c1.6_s0.05 = 0.2*read.table("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.05_seed0.txt", quote="\"")

# 0.1 noise scale
ldp_c3.2_s0.1 = 0.2*read.table("./save/NoAttacks/LDP_FL_mnist_cnn_norm3.2_scale0.1_seed0.txt", quote="\"")
ldp_c1.6_s0.1 = 0.2*read.table("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.1_seed0.txt", quote="\"")

# 0.15 noise scale
ldp_c3.2_s0.15 = 0.2*read.table("./save/NoAttacks/LDP_FL_mnist_cnn_norm3.2_scale0.15_seed0.txt", quote="\"")
ldp_c1.6_s0.15 = 0.2*read.table("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.15_seed0.txt", quote="\"")

# other noise scales..
ldp_c1.6_s0.25 = 0.2*read.table("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.25_seed0.txt", quote="\"")
ldp_c1.6_s0.5 = 0.2*read.table("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.5_seed0.txt", quote="\"")

#### CDP ####
# 0.05 noise scale
gdp_c3.2_s0.05 = 0.2*read.table("./save/NoAttacks/GDP_mnist_cnn_norm3.2_scale0.05_seed0.txt", quote="\"")
gdp_c1.6_s0.05 = 0.2*read.table("./save/NoAttacks/GDP_mnist_cnn_norm1.6_scale0.05_seed0.txt", quote="\"")

# 0.1 noise scale
gdp_c3.2_s0.1 = 0.2*read.table("./save/NoAttacks/GDP_mnist_cnn_norm3.2_scale0.1_seed0.txt", quote="\"")
gdp_c1.6_s0.1 = 0.2*read.table("./save/NoAttacks/GDP_mnist_cnn_norm1.6_scale0.1_seed0.txt", quote="\"")

# 0.15 noise scale
gdp_c3.2_s0.15 = 0.2*read.table("./save/NoAttacks/GDP_mnist_cnn_norm3.2_scale0.15_seed0.txt", quote="\"")
gdp_c1.6_s0.15 = 0.2*read.table("./save/NoAttacks/GDP_mnist_cnn_norm1.6_scale0.15_seed0.txt", quote="\"")


for (seed in 1:4){
      # non-private
      non_private = non_private + 0.2*read.table(paste0("./save/NoAttacks/PrivateFL_mnist_cnn_seed", seed, ".txt"), quote="\"")

      # LDP
      ldp_c3.2_s0.05 = ldp_c3.2_s0.05 + 0.2*read.table(paste0("./save/NoAttacks/LDP_FL_mnist_cnn_norm3.2_scale0.05_seed", seed, ".txt"), quote="\"")
      ldp_c1.6_s0.05 = ldp_c1.6_s0.05 + 0.2*read.table(paste0("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.05_seed", seed, ".txt"), quote="\"")

      ldp_c3.2_s0.1 = ldp_c3.2_s0.1 + 0.2*read.table(paste0("./save/NoAttacks/LDP_FL_mnist_cnn_norm3.2_scale0.1_seed", seed, ".txt"), quote="\"")
      ldp_c1.6_s0.1 = ldp_c1.6_s0.1 + 0.2*read.table(paste0("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.1_seed", seed, ".txt"), quote="\"")

      ldp_c3.2_s0.15 = ldp_c3.2_s0.15 + 0.2*read.table(paste0("./save/NoAttacks/LDP_FL_mnist_cnn_norm3.2_scale0.15_seed", seed, ".txt"), quote="\"")
      ldp_c1.6_s0.15 = ldp_c1.6_s0.15 + 0.2*read.table(paste0("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.15_seed", seed, ".txt"), quote="\"")

      ldp_c1.6_s0.25 = ldp_c1.6_s0.25 + 0.2*read.table(paste0("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.25_seed", seed, ".txt"), quote="\"")
      ldp_c1.6_s0.5 = ldp_c1.6_s0.5 + 0.2*read.table(paste0("./save/NoAttacks/LDP_FL_mnist_cnn_norm1.6_scale0.5_seed", seed, ".txt"), quote="\"")

      # CDP
      gdp_c3.2_s0.05 = gdp_c3.2_s0.05 + 0.2*read.table(paste0("./save/NoAttacks/GDP_mnist_cnn_norm3.2_scale0.05_seed", seed, ".txt"), quote="\"")
      gdp_c1.6_s0.05 = gdp_c1.6_s0.05 + 0.2*read.table(paste0("./save/NoAttacks/GDP_mnist_cnn_norm1.6_scale0.05_seed", seed, ".txt"), quote="\"")

      gdp_c3.2_s0.1 = gdp_c3.2_s0.1 + 0.2*read.table(paste0("./save/NoAttacks/GDP_mnist_cnn_norm3.2_scale0.1_seed", seed, ".txt"), quote="\"")
      gdp_c1.6_s0.1 = gdp_c1.6_s0.1 + 0.2*read.table(paste0("./save/NoAttacks/GDP_mnist_cnn_norm1.6_scale0.1_seed", seed, ".txt"), quote="\"")

      gdp_c3.2_s0.15 = gdp_c3.2_s0.15 + 0.2*read.table(paste0("./save/NoAttacks/GDP_mnist_cnn_norm3.2_scale0.15_seed", seed, ".txt"), quote="\"")
      gdp_c1.6_s0.15 = gdp_c1.6_s0.15 + 0.2*read.table(paste0("./save/NoAttacks/GDP_mnist_cnn_norm1.6_scale0.15_seed", seed, ".txt"), quote="\"")
}

# CDP Plots
jpeg("./plotting/NoAttack_cdp.jpg")
plot(0:20, non_private$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
# 0.05 noise
lines(0:20, gdp_c3.2_s0.05$V1, col="green", lty=1)
lines(0:20, gdp_c1.6_s0.05$V1, col="green", lty=2)
# 0.1 noise
lines(0:20, gdp_c3.2_s0.1$V1, col="red", lty=1)
lines(0:20, gdp_c1.6_s0.1$V1, col="red", lty=2)
# 0.15 noise
lines(0:20, gdp_c3.2_s0.15$V1, col="blue", lty=1)
lines(0:20, gdp_c1.6_s0.15$V1, col="blue", lty=2)
title(main="Central Differential Privacy")
legend("bottomright", c("Non-Private",
                        expression(paste(sigma, "=0.05, C=3.2")), expression(paste(sigma, "=0.05, C=1.6")),
                        expression(paste(sigma, "=0.10, C=3.2")), expression(paste(sigma, "=0.10, C=1.6")),
                        expression(paste(sigma, "=0.15, C=3.2")), expression(paste(sigma, "=0.15, C=1.6"))),
        col = c("black", "green", "green", "red", "red", "blue", "blue"), lty=c(1,1,2,1,2,1,2))
dev.off()






# LDP Plots
jpeg("./plotting/NoAttack_ldp_check.jpg")
plot(0:20, non_private$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
# 0.05 noise
lines(0:20, ldp_c3.2_s0.05$V1, col="green", lty=1)
lines(0:20, ldp_c1.6_s0.05$V1, col="green", lty=2)
# 0.1 noise
lines(0:20, ldp_c3.2_s0.1$V1, col="red", lty=1)
lines(0:20, ldp_c1.6_s0.1$V1, col="red", lty=2)
# 0.15 noise
lines(0:20, ldp_c3.2_s0.15$V1, col="blue", lty=1)
lines(0:20, ldp_c1.6_s0.15$V1, col="blue", lty=2)
title(main="Local Differential Privacy")
legend("bottomright", c("Non-Private",
                        expression(paste(sigma, "=0.05, C=3.2")), expression(paste(sigma, "=0.05, C=1.6")),
                        expression(paste(sigma, "=0.10, C=3.2")), expression(paste(sigma, "=0.10, C=1.6")),
                        expression(paste(sigma, "=0.15, C=3.2")), expression(paste(sigma, "=0.15, C=1.6"))),
        col = c("black", "green", "green", "red", "red", "blue", "blue"), lty=c(1,1,2,1,2,1,2))
dev.off()

