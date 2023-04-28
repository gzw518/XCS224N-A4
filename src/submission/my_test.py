# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#
# # 假设我们有一个批次的序列数据，包含3个序列，长度分别为5、3、2
# seqs = [torch.randn(5), torch.randn(3), torch.randn(2)]
# print("original seq",seqs)
# lengths = [len(seq) for seq in seqs]
# print(lengths)
#
#
# # 对序列进行padding操作
# padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
# print("before pack_padded",padded_seqs)
# # 将padding后的序列数据打包成一个新的序列
# packed_seqs = pack_padded_sequence(padded_seqs, lengths, batch_first=True)
# print("after pack_padded",packed_seqs)

#
# # 假设我们有一个LSTM模型
# lstm = nn.LSTM(input_size=10, hidden_size=20)
#
# # 初始化LSTM的状态
# hx = torch.randn(1, len(seqs), 20)
# cx = torch.randn(1, len(seqs), 20)
#
# # 将打包后的序列数据输入到LSTM中进行处理
# output, (hx, cx) = lstm(packed_seqs, (hx, cx))
#
# # 解包LSTM的输出数据
# output, _ = pad_packed_sequence(output, batch_first=True)
#
# # 打印最终的输出和隐藏状态
# print(output)
# print(hx)







# import torch
#
# x = torch.randn(2, 3)  # 创建一个形状为 (2, 3) 的张量
# y = x.unsqueeze(0)     # 在第 0 个维度上增加一个维度
# z = y.unsqueeze(0)
# print(y.shape)         # 输出: torch.Size([1, 2, 3])
# print(y)
# print(z.shape)
# print(z)



# import torch
#
# # Define some example tensors
# batch_size, n, m, p, q = 2, 3, 4, 5, 6
# x = torch.randn(batch_size, n, m)
# y = torch.randn(batch_size, m, p)
#
# # Compute the matrix product using torch.bmm
# z = torch.bmm(x, y)
#
# # Print the shapes of the input and output tensors
# print("x shape:", x.shape)
# print("y shape:", y.shape)
# print("z shape:", z.shape)
#
# print("x value:", x)
# print("y value:", y)
# print("z value:", z)


# import torch
#
# # 生成两个形状相同的张量
# x = torch.randn(2, 3)
# y = torch.randn(2, 3)
#
# # 沿着第0维进行拼接
# z0 = torch.cat((x, y), dim=0)
# print(f"拼接结果 (dim=0):\n{z0}\n")
#
# # 沿着第1维进行拼接
# z1 = torch.cat((x, y), dim=1)
# print(f"拼接结果 (dim=1):\n{z1}")



# import torch
#
# # 初始化隐藏状态和单元状态
# num_layers = 2
# batch_size = 3
# hidden_size = 4
# init_decoder_hidden = torch.zeros(num_layers, batch_size, hidden_size)
# init_decoder_cell = torch.zeros(num_layers, batch_size, hidden_size)
#
# # 将隐藏状态和单元状态拼接在一起，用于初始化 LSTM/GRU 模型的隐藏状态
# dec_init_state = (init_decoder_hidden, init_decoder_cell)
#
# # 打印初始化的隐藏状态
# print(dec_init_state[0].shape)
# print(dec_init_state[1].shape)
#
# print(dec_init_state[0])
#
# print(dec_init_state[1])





# import torch
#
# # 假设您有一个形状为 (src_len, b, h*2) 的张量
# src_len, b, h = 10, 5, 4
# original_tensor = torch.randn(src_len, b, h * 2)
#
# # 使用permute方法改变维度顺序
# permuted_tensor = original_tensor.permute(1, 0, 2)
#
# # 现在，permuted_tensor的形状为 (b, src_len, h*2)
# print(f"Original shape: {original_tensor.shape}")
# print(f"Permuted shape: {permuted_tensor.shape}")
#





# sents_padded = [[1,2,3],[4,5],[6]]
# for i in range(len(sents_padded)):
#     print(len(sents_padded[i]))
#
# # maxLen = 0
# # for i in range(len(sents_padded)):
# #     if len(sents_padded[i]) > maxLen:
# #         maxLen = len(sents_padded[i])
#
# maxLen = max([len(sent) for sent in sents_padded])
#
# print("maxLen:")
# print(maxLen)
#
# # for i in range(len(sents_padded)):
# #     if len(sents_padded[i]) < maxLen:
# #         for j in range(len(sents_padded[i]),maxLen):
# #             sents_padded[i].append(0)
#
# for sent in sents_padded:
#     while len(sent) < maxLen:
#         sent.append(0)
#
# print(sents_padded)