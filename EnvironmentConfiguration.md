## <a name="Environment-Configuration"></a> Environment Configuration

Choice of Working Environments: I recommend using *VScode* with remote *WSL* and *ssh on linux servers* for projects, while using *JupyterLab* for prototyping.

Manage of python packages: I recommend *Anaconda* for this task. 

Manage of version control and collaboration: Use *git* for personal projects, Use *Github* for open projects, *GitLab* for industrial projects.

Google or bing these keywords to find revelvant tutorials. This will be the last reminder of such actions.

### Conda environment quick set up with python Jupyter Lab

Save environment to a text ﬁle.

install for example, jupyterlab.

```bash
conda install -c conda-forge jupyterlab
```

```bash
conda list --explicit > bio-env.txt
```

Create environment from a text ﬁle.

```bash
conda env create --file bio-env.txt nameHere
```

Using Jupyter Lab with a bash shell and [vim-Plugin](https://github.com/jwkvam/jupyterlab-vim) is my current setup.

### Local Developing Environment setup: VScode with WSL

WSL is shorthand for Windows Subsystem for Linux, essentially, you can use linux working environment on Windows with ease. Install WSL according to this [guide](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

Then you start to work with VScode and WSL following this [guide](https://code.visualstudio.com/remote-tutorials/wsl/getting-started).

Remember to set up python3 environment according to this [guide](https://code.visualstudio.com/docs/languages/python). And the links in the guide above. Notice that, anaconda, jupyter notebook are all supported in VScode as well.

### Install anaconda on WSL inside VScode

Open bash terminal:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sha256sum filename  ### check the file is secure
bash Anaconda3--2019.10-Linux-x86_64.sh ## install it

conda update conda ## update conda to the newest version
```

### Getting access to Remote destinations: ssh

Get to know what is a [ssh](https://www.ssh.com/ssh/key) key.
Put it simply, you generate public and private key on your computer, you give ssh public key on remote machines which you have access for. Then you can log in to these remote destinations securely.

Following this [guide](https://help.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account) to add a ssh public key to your github account.

On Mac VScode bash, when you try to connect to a host with alias instead of IP, you may run into the problem of not able to connect to the remote. Now you need to edit host file on Mac and /etc/hosts in WSL to add IP address and Domain Name.

### Work on remote destinations using VScode

Follow this [guide](https://code.visualstudio.com/docs/remote/remote-overview).
