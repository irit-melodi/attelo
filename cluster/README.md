## Using the harness on a SLURM cluster

The configuration we assume is that you have a team of users working
on different aspects of the system in a shared space. One of the users
(eg. Eric) should be designated “local admin”.

## Initialisation 1 (local admin)

1. Upload the corpora to a shared space

2. Install miniconda (Python 2.7 version for now; we think Python 3 is
   doable too but it'll probably involve some tweaks to educe…)

## Initialisation 2 (you)

1. Put the following in your .bashrc

   ```
   MINICONDA_DIR=EDIT_THIS   # <== get from local admin
   PROJECT_DIR=EDIT_THIS # <== get from local admin
   export PATH=$MINICONDA_DIR/bin:$PATH
   ```

   (PROJECT_DIR is not strictly speaking needed; it's just a handy
   way to refer to the shared space below)

2. Start bash (it's tcsh by default for us; I couldn't figure out
   how to change my login shell)

3. Create and activate your personal conda environment
   (HINT: is conda in your path? It should be after steps 1 and 2)

      ```
      conda create -n irit-rst-dt-$USER scipy pip
      source activate irit-rst-dt-$USER
      ```

4. Fetch irit-rst-dt into the shared space and link it into your
   home directory

    ```
    cd $PROJECT_DIR # or just type path in yourself
    git clone https://github.com/kowey/irit-rst-dt.git irit-rst-dt-$USER
    ln -s $PROJECT_DIR/irit-rst-dt-$USER $HOME/irit-rst-dt
    ```

5. (optional) modify the requirements.txt to point to your personal educe
   and attelo branch (HINT: you can refer to branches on GitHub
   repositories)

6. Link the RST-DT and PTB corpora (path provided by local admin) in
   (see main README)

7. Run the usual install

   ```
   pip install -r requirements.txt
   ```

8. Set up your cluster scripts (replace vim with your favourite text
   editor below). You'll need to plug in your email address and
   absolute paths appropriate to your cluster

   ```
   for i in env gather.script; do
       cp cluster/$i.example cluster/$i
   done
   vim cluster/env
   vim cluster/gather.script
   ```

## Using the cluster scripts

1. Launch feature extraction

```
bash
cd irit-rst-dt
sbatch cluster/gather.script
```

2. Look out for a slurm-?????.out file. Check its contents ocassionally.
   Does feature extraction seem to be properly running? How about a nice
   coffee then?

3. Launch the experiment (I assume here you'd been automatically logged
   out)

```
bash
cd irit-rst-dt
chmod u+x cluster/go
cluster/go
```

## Hints

* the `cluster/go` script can accept arguments for `irit-rst-dt
  evaluate` on the command line

* to monitor progress, you might run something like `watch -d -t -n 10 'echo "---- WATCH  ---"; tail -n 1 i*.out'` in your irit-rst-dt dir.  This tails all of the current log files every 10 seconds, highlighting anything that has changed
