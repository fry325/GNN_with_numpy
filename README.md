
使用numpy实现图神经网络节点分类  
implement gnn node classification with numpy  
# Theory
经过我的一番推导发现，对于任意有如下形式的GNN：  
H = A^K * X  
y_pred = softmax(H * W)  
其中A可以是任意propagation matrix  
W的梯度是：  
gradient = H^T * (y_pred - y)  
其中gradient里面的y_pred和y都经过了mask，y每一行不是training set里的节点会被置零  

I found that any node classification GNN with following structure:   
H = A^K * X  
y_pred = softmax(H * W)  
A can be any propagation matrix  
Gradient of W is:  
gradient = H^T * (y_pred - y)  
while computing gradient, y and y_pred will be masked. For every row of y and y_pred, node that do not belongs to training set will be set to zero
# Core Code
```
class NumpySGC:
    def __init__(self, n_features, n_labels):
        self.W = xavier_uniform((n_features, n_labels))
    
    def forward_propagate(self, H):
        return softmax(np.matmul(H, self.W))
    
    def backward_propagate(self, H, y, y_pred, mask, optimizer):
        gradient =  - H.T @ (y - (mask @ y_pred))
        self.W = optimizer.update(gradient, self.W)
```
# results on citation network
sample 20 nodes for each class to training set
| model  | cora_ml | citeseer | pubmed |
| ----  | ----  | ----  | ----  |
| SGC | 81.09 ± 1.13 %  |  69.46 ± 1.71 %  |  76.03 ± 1.86 %  |
| numpy_SGC | 77.33 ± 1.13 %  |   64.03 ± 2.26 %  |   -  |

sample 60% nodes for each class to training set
| model  | cora_ml | citeseer | pubmed |
| ----  | ----  | ----  | ----  |
| SGC  | 87.90 ± 1.6 % | 75.57 ± 1.49 %  | 87.16 ± 0.47 %  |
| numpy_SGC | 83.47 ± 1.44 %  |   71.64 ± 2.11 %  |   -  |