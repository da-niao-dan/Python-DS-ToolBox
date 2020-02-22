# <a name="Git"> Git </a>

There is a official git [guide](https://git-scm.com/). Use it as reference and install git according to it.

However, here are some basic git commands to get you up to speed:

```bash
git init ###initialize a git repo locally at your current folder
git remote add origin linkToGitRepoOnline ## set a remote destination to sync with your git repo
git remote set-url origin linkToGitRepoOnline ## change the url of your remote called origin. from now on, origin stands for linkToGitRepoOnline
git pull origin branchNameHere ## pull data from remote branch to local repo

# Branching
git branch newBranch master ## create a new branch from master branch
git checkout -b branchNameHere ## switch to a branch
git branch -d branchNameHere ## delete a branch
git checkout master ## switch to master branch
git merge branchNameHere ## merge branchNameHere branch with current branch

# you modify or create some files
git add * ## stage all files you modified
git commit ## Say what you changed
git commit -a -m 'changes descriptions' ## quicker, but essentially same as above

git push origin master ## push changes to a remote branch master of origin, Here origin is a remote name, master is a branch name.
git push --set-upstream remoteNameHere branchNameHere ## If you run this once, later you only need:
git push
```
