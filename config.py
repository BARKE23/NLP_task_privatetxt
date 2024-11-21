# 模型相关配置
MODEL_NAME = 'roberta-base' #预训练模型
NUM_CLASSES = 31 #分类类别数量
# 训练相关配置
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
MAX_LENGTH = 256  # 文本最大长度，用于截断或填充
# 数据路径相关配置
TRAIN_JSON_PATH = '/home/hlh/datasets/nlpdata/train.json'
VALID_JSON_PATH = '/home/hlh/datasets/nlpdata/valid.json'
TEST_TXT_PATH = '/home/hlh/datasets/nlpdata/test.txt'
LABEL_LIST_PATH = '/home/hlh/datasets/nlpdata/label_list.txt'
#RESULT_TXT_PATH ='/home/hlh/datasets/nlpdata/220242221043_houliuhui.txt'  # 保存测试结果的txt文件路径
RESULT_TXT_PATH ='/home/hlh/datasets/nlpdata/result903.txt'
CHECKPOINT_PATH = '/home/hlh/projects/nlp/checkpoints/best_model_1120-1904_lr0.0001_bs64_ep30.pth'