[官方文档](https://ipython.readthedocs.io/en/stable/interactive/magics.html)

## 常用
```Python
%load_ext autoreload # 自动重新加载更改的模块
%autoreload 2
%qtconsole # 启动和当前笔记本相同内核的 qtconsole
%connect_info # 当前笔记本链接信息
```

## Line magics
```Python
%alias	# 定义别名
%alias_magic	# 为现有的魔术命令创建别名
%autocall
%automagic	# 设置输入魔术命令时是否键入%前缀，on(1)/off(0)
%bookmark	# 管理IPython的书签系统
%cd	# 更改当前工作目录
%colors
%config
%debug
%dhist # 打印历史访问目录
%dirs # 返回当前目录堆栈
%doctest_mode
%edit
%env # 设置环境变量(无需重启)
%gui
%history
%killbgscripts
%load	# 导入python文件
%load_ext
%loadpy	# %load别名
%logoff	# 临时停止logging
%logon	# 重新开始logging
%logstart
%logstate
%lsmagic	# 列出当前可用的魔术命令。
%macro	# 定义用来重复执行的宏
%magic	# 显示魔术命令的帮助
%matplotlib	# 设置matplotlib的工作方式
%notebook
%page
%pastebin
%pdb	# 控制pdb交互式调试器的自动调用
```

## 打印相关

```Python
%pdef	# 打印任何可调用对象信息
%pdoc	# 打印对象的docstring
%pfile
%pinfo
%pinfo2
%pip	运行pip命令
%popd
%pprint	# 美化打印
%precision	# 设置美化打印时的浮点数精度
%profile	# 打印您当前活动的IPython配置文件
%prun	# 告诉你程序中每个函数消耗的时间
%psearch
%psource	# 打印对象源代码
%pushd
%pwd	# 返回当前工作路径
%pycat
%pylab	# 加载numpy、matplotlib
%quickref
%recall
%rehashx
```

## 运行相关
```Python
%reload_ext	# 通过其模块名称重新加载IPython扩展
%rerun
%reset
%reset_selective
%run
%save
%sc
%set_env	# 设置环境变量
%sx
%system
%tb
%time	# 执行Python语句或表达式的时间
%timeit
%unalias	# 移别名
%unload_ext	# 通过其模块名称卸载IPython扩展
%who	# 列出全局变量
%who_ls	 # 以排序列表的方式列出变量
%whos	 # 类似who，但给出的信息更详细
%xdel
%xmode
```

## Cell magics
在 notebook 内用不同的内核运行代码

```Python
%%bash
%%capture
%%html
%%javascript
%%js
%%latex
%%markdown
%%perl
%%pypy
%%python
%%python2
%%python3
%%ruby
%%sh
%%svg
%%writefile
```
