目前主流的调参算法包括Grid search、Random search、TPE、PSO、SMAC以及贝叶斯调参等

本文简单介绍Hyperopt自动调参框架的设计和实现

## [Hyperopt](https://github.com/hyperopt/hyperopt)

Hyperopt是python中的一个用于"分布式异步算法组态/超参数优化"的类库。使用它我们可以拜托繁杂的超参数优化过程，自动获取最佳的超参数。

广泛意义上，可以将带有超参数的模型看作是一个必然的非凸函数，因此hyperopt几乎可以稳定的获取比手工更加合理的调参结果。尤其对于调参比较复杂的模型而言，其更是能以远快于人工调参的速度同样获得远远超过人工调参的最终性能。

Hyperopt调参框架 支持Random search，和TPE（Tree of Parzen Estimators，优化后的贝叶斯自动调参，可依赖于mongoDB实现分布式调参。

### 安装
```Python
pip install hyperopt
```

Hyperopt的基本框架基于定义的最小化的目标函数，在给定的搜索空间范围内，使用Random search或者贝叶斯自动调参的算法，获取模型最佳性能的调参结果。

### 一个简单的例子
```Python
# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227}
```

### 目标函数
示例中的objective函数，即为Hyperopt自动调参的目标函数。Hyperopt自动调参或解决问题的关键就是通过搜索参数空间给定的参数，实现目标函数最小化（fmin函数），就是模型的最佳参数

### 参数空间
定义的space即为自动调参定义的参数空间，自动调参的参数范围会在参数空间中选择或遍历，Hyperopt提供的定义参数空间的类型包括：

- hp.choice：对定义的list或tuple中元素随机选择；
- hp.uniforme：定义一个连续的数值范围
- hp.randint：定义从0开始的整数范围，当不需要从0开始时，可以加常数进行自己定义
- hp.normal：定义一个正态分布的连续数组
- 其他 hp.qnormal，hp.lognormal，hp.qlognormal，hp.quniform，hp.loguniform，hp.qloguniform 其他数据分布或是添加常数改变数值的步长或变化趋势

### 自动调参算法
fmin(objective, space, algo=tpe.suggest, max_evals=100)

- algo=tpe.suggest意思时使用tpe的自动调参策略
- 在space调参空间里面，针对目标函数objective进行自动寻优调参

### 结合sklearn实现的随机森林的交叉验证自动调参
```Python
def hyperopt_fun(X, y,params):
    '''
    Hyperopt的目标函数，调参优化的依据
    params:params,Hyperopt.hp,参数空间，调参将根据参数空间进行优化
    '''
    params=argsDict_tranform(params)
    alg=RandomForestRegressor(**params)
    metric = cross_val_score(
        alg,
        X,
        y,
        cv=10,scoring="neg_mean_squared_error")
    return min(-metric)
    def hyperopt_space():
        import hyperopt.pyll
        from hyperopt.pyll import scope
        space= {
                'n_estimators':hp.randint("n_estimators_RF", 300),
                'max_depth':hp.randint("max_depth_RF", 35),
                'min_samples_split':hp.uniform('min_samples_split_RF',0.4,0.7),
                'min_samples_leaf':hp.randint('min_samples_leaf_RF',300),
                'min_weight_fraction_leaf':hp.uniform('min_weight_fraction_leaf_RF',0,0.5),
                'max_features':hp.uniform('max_features_RF',0.5,1.0),
                'oob_score':True, 
                'n_jobs':-1, 
                'random_state':2019
            }
        return space
    def argsDict_tranform(argsDict,isPrint=False,best=False):
        if best:
            ### 对获取到的最后调优结果进行转换参数

            argsDict['n_estimators']=argsDict.pop('n_estimators'+'_%s'%t)+1
            argsDict['max_depth']=argsDict.pop('max_depth'+'_%s'%t)+7
            argsDict['min_samples_leaf']=argsDict.pop('min_samples_leaf'+'_%s'%t)+100
            argsDict['min_samples_split']=argsDict.pop('min_samples_split'+'_%s'%t)
            argsDict['min_weight_fraction_leaf']=argsDict.pop('min_weight_fraction_leaf'+'_%s'%t)
            argsDict['max_features']=argsDict.pop('max_features'+'_%s'%t)
        else:
            ###调参过程中，对于采样空间的处理，例如有些参数不能为0之类的情况
            argsDict['n_estimators']=argsDict['n_estimators']+1
            argsDict['max_depth']=argsDict['max_depth']+7
            argsDict['min_samples_leaf']=argsDict['min_samples_leaf']+100
        if isPrint:
            print(argsDict)
        else:
            pass
        return argsDict
    def hyperopt_hpo(max_evals=100):
        algo=partial(tpe.suggest,n_startup_jobs=1)
        space=hyperopt_space()
        trials=Trials()
        best=fmin(hyperopt_fun,space,algo=algo,trials=trials,max_evals=max_evals, pass_expr_memo_ctrl=None)
        print(best)
        best_t=argsDict_tranform(best,best=True)
        return best_t
```
---
## Google Vizier
只是Google CloudML平台提供的调参服务，用户把Python代码上传上去，定义一个HyperparameterSpec，云平台就会使用调参算法并行训练并且选择效果最优的超参组合和模型。

在Google内部，Vizier不仅提供调参服务给Google Cloud的服务，面向跟底层还提供了批量获取推荐超参、批量更新模型结果、更新和调试调参算法以及Web控制台等功能。

## Microsoft NNI
NNI（Neural Network Intelligence）是微软开源的自动机器学习工具包

项目地址：[Microsoft/nni(https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni)] 

它本地、远程服务器和云端通过不同的调参算法来寻找最优的神经网络架构和超参数

自带多种调参算法：TPE, Random, Anneal, Evolution, BatchTuner等,可以通过webUI界面查看和分析实验结果

## Advisor
基于Vizier论文实现的调参服务，同时也集成了NNI的接口使用特点，当然还有最流行的BayesianOptimization等算法实现

开源地址:[tobegit3hub/advisor](https://github.com/tobegit3hub/advisor)

但是目前最新的版本只支持到python3.5，在应用中还是比较受限
