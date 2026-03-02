# 流程
- 下载git，选择Windows x86最新维护版本，使用vim作为git的默认编辑器（default editor）
- 检查是否下载成功：
```
$ git --version
$ git config --list
```

- 设置用户个人信息：
```
$ git config --global user.name "mirror"
$ git config --global user.email 3480839084@qq.com
```

- 生成ssh密钥并添加到github
```
$ ssh-keygen -t rsa -C "3480839084@qq.com"
后续默认回车
```

- 检查github连接：
```
$ ssh -T git@github.com
键入yes连接成功
```

但此时出现问题，报错：ssh: connect to host github.com port 22: Connection refused
查找网上经验，找到https://zhuanlan.zhihu.com/p/521340971，首先尝试更换端口443连接
```
$ ssh -T -p 443 git@ssh.github.com
```

发现可以连接
在~/.ssh文件夹中创建无后缀config文件，打开后键入：
```
Host github.com
  Hostname ssh.github.com
  Port 443
```
之后$ ssh -T git@github.com命令后回应：Hi Mirror99m! You've successfully authenticated, but GitHub does not provide shell access.

