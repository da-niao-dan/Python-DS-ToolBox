
# <a name="Linux-and-Bash-shells">Linux and Bash shells</a> 

Google *Basic Bash Commands* for pwd, cp, cd, ls, cat, vim, nano, >, mv, sudo, apt update, apt upgrade, apt intall, man, du, df  etc...

## Login to a server

```bash
ssh -A yourAccountName@ipAddressOrHostName
```

## Manage your ssh identity

activate your ssh-agent

```bash
eval $(ssh-agent -s)

```

add a ssh identity, for example your private key is of name id_rsa

```bash
ssh-add id_rsa
```

## change permission status of a file

```bash
chmod someCode yourFile
```

someCode=400 makes it non-writable by your own user.

someCode=600 allows owner read-write not just read.

someCode=700 allows owner to read write and execute a file/folder.

**If you have problem with ssh logins, use the following command:**

```bash
chmod 700 -R ~/.ssh
chmod 400 ~/.ssh/id_rsa   ## assuming id_rsa is your private key
```

## Change Ownership of a file

```bash
chown new-owner  filename
```

new-owner: Specifies the user name or UID of the new owner of the file or directory.  
filename: Specifies the file or directory.

## File Transfer

Copy LocalFile to your remoteDestination

```bash
scp LocalFile RemoteDestinationFolder
```

Sometimes File Transfer may fail because perssion denied. You need to change ownership of the file.

## Activate and shut down your root power* (Caution, don't do this if you don't know why root power is dangerous.)

Activate: use `sudo su -`
Shut down: use `exit` or `logout`

## Check processes

check all processes

```bash
ps aux
```

check processes with keyword, for example: *agent*.

```bash
ps aux | grep agent

#Alternatively, directly get process ID
pgrep agent
```

kill process

```bash
kill taskID
```

## Getting file from web on Linux Server

First, install *wget* using  `yum install wget` or `sudo apt-get install wget`.

Then type:

```bash
wget yourUrl
```

One tip for getting url from a masked hyperlink like [this] on graphical user interface:
right click the text and select 'Copy link address'.

## compress and decompress files

### zip

Although tar is the mainstream, sometimes you need to zip files.

Check this [guide](https://linuxize.com/post/how-to-zip-files-and-directories-in-linux/).


### tar files

List the contents of a tar file:

```bash
tar -tvf archive.tar
```

Extract the contents of a tar file:

```bash
tar -xf archive.tar
```

Extract all the contents of a gzipped tar file:

```bash
tar -xzvf archive.tar.gz
```

Extract one file from a tar file:

```bash
tar -xvf archive.tar targetFileName
```

Extract some files specified by a particular format from a tar file:

```bash
tar -xvf archive.tar --wildcards '*2019.sh'
```

Create an tar Archive for a folder:

```bash
tar -cf archive.tar mydir/
```

Create an gzipped tar Archive for a folder :

```bash
tar -czf archive.tar.gz mydir/
```

### gzip files

decompress a file

```bash
gzip -d someFile.gz
```

compress a file
use option 1 to 9, 1 is for maximum compression at the slowest speed, 9 is for minimum compression at the fastest speed.

```bash
gzip -1 someFile
gzip -2 someFile
gzip -9 someFile
```

## Adding a PATH variable

Sometimes when you install a new software, you need to add a new PATH variable to your environment so that you can call the new software easily.

First you can run the following two lines:

```bash
PATH=YourPathToTheFolderOfTheNewSoftware:$PATH
export PATH
```

Then you can include these two lines in your ~/.bashrc file.

Sometimes you messed up the PATH variable by messed up ~/.bashrc file, (you find yourself cannot use ls or vim commands), relax, just run

```bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
```

and debug ~/.bashrc file using vim.

```bash
vim ~/.bashrc
```

## Learn to use vim

Open bash and type:

```bash
vimtutor
```

## file editing

modify table using cut.

```bash

cut -d ' ' -f 1-10 orginal_filename > filename.
```

## crontab

### run python script in crontab

*VERY IMPORTANT!*

Assume you have multiple versions of python installed on your computer.
The first step is to locate your python executable file.
Now activate the environment you want to use.
Type:

```bash

which python
```

Copy and paste the path, and run python in crontab like this:

```bash
* * * * * nohup copiedPathToPython YourAbsolutePathToScript  >> /home/yourUserName/cron_lab.log 2>&1
```

### run python in crontab with characters other than English

For example, if you have Chinese character in your python script, you need to
include `# -*- coding: utf-8 -*-` at the top of python script.

## Set your sudo password feedback visible in *

```bash
sudo visudo
```

find the line with 'Defaults env_reset', change it to 'Defaults env_reset,pwfeedback'.

save the file.
Now refresh and test.

```bash
sudo -k
sudo ls
```

