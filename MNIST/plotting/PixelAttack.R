# Title     : T
# Objective : Plot
# Created by: spyros
# Created on: 2021-04-14
getwd()


no_defenseTA = 0.2 * read.table("./save/PixelAttack/TestAcc/NoDefense_mnist_cnn_attackers2_seed0.txt", quote="\"")
no_defenseBA = 0.2 * read.table("./save/PixelAttack/BackdoorAcc/NoDefense_mnist_cnn_attackers2_seed0.txt", quote="\"")


LDP_TA = 0.2 * read.table("./save/PixelAttack/TestAcc/LDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed0.txt", quote="\"")
LDP_BA = 0.2 * read.table("./save/PixelAttack/BackdoorAcc/LDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed0.txt", quote="\"")

CDP_TA = 0.2 * read.table("./save/PixelAttack/TestAcc/GDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed0.txt", quote="\"")
CDP_BA = 0.2 * read.table("./save/PixelAttack/BackdoorAcc/GDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed0.txt", quote="\"")

for (seed in 1:4){

      # No Defense
      no_defenseTA = no_defenseTA + 0.2 * read.table(paste0("./save/PixelAttack/TestAcc/NoDefense_mnist_cnn_attackers2_seed", seed, ".txt"), quote="\"")
      no_defenseBA = no_defenseBA + 0.2 * read.table(paste0("./save/PixelAttack/BackdoorAcc/NoDefense_mnist_cnn_attackers2_seed", seed, ".txt"), quote="\"")

      # LDP
      LDP_TA = LDP_TA + 0.2 * read.table(paste0("./save/PixelAttack/TestAcc/LDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed", seed, ".txt"), quote="\"")
      LDP_BA = LDP_BA + 0.2 * read.table(paste0("./save/PixelAttack/BackdoorAcc/LDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed", seed, ".txt"), quote="\"")

      # CDP
      CDP_TA = CDP_TA + 0.2 * read.table(paste0("./save/PixelAttack/TestAcc/GDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed", seed, ".txt"), quote="\"")
      CDP_BA = CDP_BA + 0.2 * read.table(paste0("./save/PixelAttack/BackdoorAcc/GDP_mnist_cnn_clip2.2_scale0.1_attackers2_seed", seed, ".txt"), quote="\"")
}


# PLOT TEST ACCURACY
jpeg("./plotting/PixelAttack_TA.png")
plot(0:20, no_defenseTA$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
lines(0:20, CDP_TA$V1, col="green")
lines(0:20, LDP_TA$V1, col="red")
title(main="Main Task")
legend("bottomright", c("None", "LDP", "CDP"), col = c("black", "red", "green"), lty=c(1,1,1), title = "Defense")
dev.off()


# PLOT BACKDOOR ACCURACY
jpeg("./plotting/PixelAttack_BA.png")
plot(0:20, no_defenseBA$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
lines(0:20, CDP_BA$V1, col="green")
lines(0:20, LDP_BA$V1, col="red")
title(main="Backdoor Task")
legend("bottomright", c("None", "LDP", "CDP"), col = c("black", "red", "green"), lty=c(1,1), title = "Defense")
dev.off()

