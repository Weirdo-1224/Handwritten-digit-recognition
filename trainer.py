# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from optimizer import *


class Trainer:
    """进行神经网络的训练的类
       evaluate_sample_num_per_epoch: 这个参数用于指定每个训练周期（epoch）结束后，在验证集上评估（测试）模型性能时使用的样本数量。
       通常在训练神经网络时，我们不需要在每个 epoch 结束后都使用完整的验证集进行性能评估，因为这可能会增加计算开销。所以，可以通过设置
       evaluate_sample_num_per_epoch 来指定每个 epoch 结束后用多少个样本来评估模型。如果设置为 None，则表示在每个 epoch 结束后
       使用整个验证集。这个参数可以用来加快训练速度和减少计算资源的消耗。

       verbose: 这个参数用于控制训练过程中的输出信息。如果设置为 True，那么在每个 epoch 结束后，训练过程会输出一些信息，如当前 epoch 数、
       训练集上的损失、验证集上的准确度等。这些信息有助于监视训练过程。如果设置为 False，训练过程将不输出这些信息，可以减少屏幕上的输出量。
    """

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        """
                   Parameters:
                - network: 要训练的神经网络模型的实例。
                - x_train: 训练数据。
                - t_train: 训练数据的标签。
                - x_test: 验证数据。
                - t_test: 验证数据的标签。
                - epochs: 训练的总周期数。
                - mini_batch_size: 每个小批量的样本数量。
                - optimizer: 优化算法的选择，如 'sgd'、'momentum'、'nesterov' 等。
                - optimizer_param: 一个包含优化算法参数的字典，如学习率 'lr'。
                - evaluate_sample_num_per_epoch: 每个 epoch 结束后用于验证的样本数量，None 表示整个验证集。
                - verbose: 控制是否输出训练过程中的详细信息。
               """

        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
                                'adagrad': AdaGrad, 'rmsprpo': RMSprop, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        # 随机选择一批次的数据索引
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        # 根据选中的索引获取当前批次的训练数据和标签
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 计算当前批次数据的梯度
        grads = self.network.gradient(x_batch, t_batch)
        # 使用优化器更新网络参数
        self.optimizer.update(self.network.params, grads)

        # 计算当前批次的损失并添加到训练损失列表
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        # 如果verbose为True，打印当前训练损失
        if self.verbose: print("train loss:" + str(loss))

        # 如果当前迭代次数是每个epoch的整数倍，则进行评估
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            # 评估训练集和测试集
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            # 如果设置了评估样本数量，则使用部分样本进行评估
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            # 计算训练集和测试集的准确率
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            # 将准确率添加到相应的列表
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            # 如果verbose为True，打印当前epoch的训练和测试准确率
            if self.verbose: print(
                "=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(
                    test_acc) + " ===")
        # 增加当前迭代次数
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

