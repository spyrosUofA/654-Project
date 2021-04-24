# Title     : T
# Objective : Plot
# Created by: spyros
# Created on: 2021-04-14
getwd()

no_defense = read.table("./save/RandomAttack/NoDefense_iid_mnist_cnn_attackers1_seed1.txt", quote="\"")
GDP = read.table("./save/RandomAttack/GDP_iid_mnist_cnn_clip3.2_scale0.15_attackers1_seed1.txt", quote="\"")

for (seed in 1:4){
      no_defense = no_defense + read.table(paste0("./save/RandomAttack/NoDefense_iid_mnist_cnn_attackers1_seed", seed, ".txt"), quote="\"")
      GDP = GDP + read.table(paste0("./save/RandomAttack/GDP_iid_mnist_cnn_clip3.2_scale0.15_attackers1_seed", seed, ".txt"), quote="\"")
}


no_defense = no_defense / 5
GDP = GDP / 5


jpeg("./plotting/RandomAttack_iid.jpg")
plot(0:20, no_defense$V1, xlab = "Communication Round", ylab = "Test Accuracy", ylim = c(0,1), type="l")
lines(0:20, GDP$V1, col="blue")
title(main="Central Differential Privacy")
legend("bottomright", c("Non-Private FL",
                        expression(paste(sigma, "=0.15, C=3.2"))),
        col = c("black", "blue"), lty=c(1,1), cex=0.75)
dev.off()