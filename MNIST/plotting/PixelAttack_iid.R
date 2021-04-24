# Title     : T
# Objective : Plot
# Created by: spyros
# Created on: 2021-04-14
getwd()


no_defenseTA = 0.2 * read.table("./save/PixelAttack/TestAcc/iid_NoDefense_mnist_cnn_attackers2_seed0.txt", quote="\"")
no_defenseBA = 0.2 * read.table("./save/PixelAttack/BackdoorAcc/iid_NoDefense_mnist_cnn_attackers2_seed0.txt", quote="\"")

CDP_1.6_TA = 0.2 * read.table("./save/PixelAttack/TestAcc/iid_GDP_mnist_cnn_clip1.6_scale0.15_attackers2_seed0.txt", quote="\"")
CDP_1.6_BA = 0.2 * read.table("./save/PixelAttack/BackdoorAcc/iid_GDP_mnist_cnn_clip1.6_scale0.15_attackers2_seed0.txt", quote="\"")

CDP_3.2_TA = 0.2 * read.table("./save/PixelAttack/TestAcc/iid_GDP_mnist_cnn_clip3.2_scale0.15_attackers2_seed0.txt", quote="\"")
CDP_3.2_BA = 0.2 * read.table("./save/PixelAttack/BackdoorAcc/iid_GDP_mnist_cnn_clip3.2_scale0.15_attackers2_seed0.txt", quote="\"")

CDP_8_BA = 0.2 * read.table("./save/PixelAttack/BackdoorAcc/iid_GDP_mnist_cnn_clip8_scale0.05_attackers2_seed0.txt", quote="\"")


for (seed in 1:4){

      # No Defense
      no_defenseTA = no_defenseTA + 0.2 * read.table(paste0("./save/PixelAttack/TestAcc/iid_NoDefense_mnist_cnn_attackers2_seed", seed, ".txt"), quote="\"")
      no_defenseBA = no_defenseBA + 0.2 * read.table(paste0("./save/PixelAttack/BackdoorAcc/iid_NoDefense_mnist_cnn_attackers2_seed", seed, ".txt"), quote="\"")

      # CDP
      CDP_1.6_TA = CDP_1.6_TA + 0.2 * read.table(paste0("./save/PixelAttack/TestAcc/iid_GDP_mnist_cnn_clip1.6_scale0.15_attackers2_seed", seed, ".txt"), quote="\"")
      CDP_1.6_BA = CDP_1.6_BA + 0.2 * read.table(paste0("./save/PixelAttack/BackdoorAcc/iid_GDP_mnist_cnn_clip1.6_scale0.15_attackers2_seed", seed, ".txt"), quote="\"")

      CDP_3.2_TA = CDP_3.2_TA + 0.2 * read.table(paste0("./save/PixelAttack/TestAcc/iid_GDP_mnist_cnn_clip3.2_scale0.15_attackers2_seed", seed, ".txt"), quote="\"")
      CDP_3.2_BA = CDP_3.2_BA + 0.2 * read.table(paste0("./save/PixelAttack/BackdoorAcc/iid_GDP_mnist_cnn_clip3.2_scale0.15_attackers2_seed", seed, ".txt"), quote="\"")

      CDP_8_BA = CDP_8_BA + 0.2 * read.table(paste0("./save/PixelAttack/BackdoorAcc/iid_GDP_mnist_cnn_clip8_scale0.05_attackers2_seed", seed, ".txt"), quote="\"")
}


# PLOT TEST ACCURACY
jpeg("./plotting/PixelAttack_TA_iid.png")
plot(0:20, no_defenseTA$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
lines(0:20, CDP_1.6_TA$V1, col="blue", lty=2)
lines(0:20, CDP_3.2_TA$V1, col="blue")
title(main="Main Task")
legend("right", c("None", expression(paste(sigma, "=0.15, C=3.2")), expression(paste(sigma, "=0.15, C=1.6"))),
        col = c("black", "blue", "blue"), lty=c(1,1,2), title = "Defense")
dev.off()


# PLOT BACKDOOR ACCURACY
jpeg("./plotting/PixelAttack_BA_iid2.png")
plot(0:20, no_defenseBA$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
lines(0:20, CDP_1.6_BA$V1, col="blue", lty=2)
lines(0:20, CDP_3.2_BA$V1, col="blue")
lines(0:20, CDP_8_BA$V1, col="red")
title(main="Backdoor Task")
legend("right", c("None", expression(paste(sigma, "=0.15, C=3.2")), expression(paste(sigma, "=0.15, C=1.6"))),
        col = c("black", "blue", "blue"), lty=c(1,1,2), title = "Defense")
dev.off()

