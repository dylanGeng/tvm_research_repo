# Git 使用方法总结

## 1. 常用Git命令

- git add
- git commit
- git push

## 2. [git add](https://www.cnblogs.com/skura23/p/5859243.html)

> `git add -A`和`git add .`以及`git add -u`在功能上看似相近，但还是存在一点差别

- git add .: 它会监控工作区的状态树，使用它会把工作时的**所有变化提交**到暂存区，包括文件内容修改(modified)以及新文件(new)，但不包括被删除的文件。
- git add -u: 它仅监控已经被add的文件（即tracked file），它会将修改的文件提交到暂存区。add -u不会提交新文件(untracked file)。(git add --update的缩写)
- git add -A: 是上面两个功能的合集(git add --all的缩写)

## 3. [git commit](http://www.cnblogs.com/qianqiannian/p/6005628.html)

> git commit 主要是将暂存区里的改动给提交到本地的版本库。每次使用git commit命令我们都会在本地版本库生成一个40位的哈希值，这个哈希值也叫commit-id。
> commit-id在版本回退的时候是非常有用的，它相当于一个快照，可以在未来的任何时候通过与git reset的组合命令回到这里。
- git commit -m "message"

## 4. [git push](https://www.yiibai.com/git/git_push.html)

> [git push](https://www.cnblogs.com/qianqiannian/p/6008140.html) 命令用于将本地分支的更新，推送到远程主机。它的格式与`git pull`命令相似。
```shell
git push <远程主机名> <本地分支名>:<远程分支名>
git.exe push --progress "origin" develop:develop
```

## 5. [git分支切换](https://blog.csdn.net/u014540717/article/details/54314126)
> git 一般有很多分支，我们clone到本地的时候一般是master分支，如何切换到其他分支上？主要使用命令如下：
- git branch -a: 查看远程分支
```
~/mxnet$ git branch -a
* master
  remotes/origin/HEAD -> origin/master
  remotes/origin/master
  remotes/origin/nnvm
  remotes/origin/piiswrong-patch-1
  remotes/origin/v0.9rc1
```
- git branch: 查看本地分支
```
~/mxnet$ git branch
* master
```
- git checkout -b develop origin/develop: 切换分支develop
```
$ git checkout -b v0.9rc1 origin/v0.9rc1
Branch v0.9rc1 set up to track remote branch v0.9rc1 from origin.
Switched to a new branch 'v0.9rc1'

＃已经切换到v0.9rc1分支了
$ git branch
  master
* v0.9rc1

＃切换回master分支
$ git checkout master
Switched to branch 'master'
Your branch is up-to-date with 'origin/master'.
```

## 6. [git fetch和git pull的区别](https://blog.csdn.net/hudashi/article/details/7664457)

> Git中从远程的分支获取最新的版本到本地有2个常用命令：

- git fetch: 相当于是从远程获取最新的版本到本地，不会自动merge
- git pull: 相当于是从远程获取最新版本并merge到本地



>下述命令的含义：首先从远程的origin的master主分支下载最新的版本到origin/master分支上，然后比较本地的master分支和origin/master分支的差别，最后进行合并。

```shell
git fetch origin master
git log -p master..origin/master
git merge origin/master
```

> 上述过程可以用更清晰的方式来进行：

```shell
git fetch origin master:tmp #从远程获取最新的版本到本地的test分支上
git diff tmp 
git merge tmp #再进行比较合并
```

> git pull origin master `命令相当于`git fetch`和`git merge`
>
> 在实际使用中，git fetch更安全一些。因为在merge前，我们可以查看更新情况，然后再决定是否合并。
```shell
git.exe pull --progress -v --no-rebase "origin"
```

## 7. [git 远程仓库的使用](https://git-scm.com/book/zh/v1/Git-%E5%9F%BA%E7%A1%80-%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%E7%9A%84%E4%BD%BF%E7%94%A8)
> 要参与任何一个 Git 项目的协作，必须要了解该如何管理远程仓库。远程仓库是指托管在网络上的项目仓库，可能会有好多个，其中有些你只能读，另外有些可以写。同他人协作开发某个项目时，需要管理这些远程仓库，以便推送或拉取数据，分享各自的工作进展。 管理远程仓库的工作，包括添加远程库，移除废弃的远程库，管理各式远程库分支，定义是否跟踪这些分支，等等。本节我们将详细讨论远程库的管理和使用。

## 8. [git remote命令](https://www.yiibai.com/git/git_remote.html)
> git remote命令管理一组跟踪的存储库。

> 要参与任何一个 Git 项目的协作，必须要了解该如何管理远程仓库。远程仓库是指托管在网络上的项目仓库，可能会有好多个，其中有些你只能读，另外有些可以写。同他人协作开发某 个项目时，需要管理这些远程仓库，以便推送或拉取数据，分享各自的工作进展。管理远程仓库的工作，包括添加远程库，移除废弃的远程库，管理各式远程库分支，定义是否跟踪这些分支等等。
