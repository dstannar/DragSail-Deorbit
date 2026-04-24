# Download Java
1.  Navigate Oracle's [Java JDK](https://www.oracle.com/java/technologies/javase/jdk23-archive-downloads.html) and download the installer appropriate for your operating system.
2.  Run the installer.
3.  Keep default settings.
3.  Click **Install**.

# Download Anaconda 
1.  Go to [Anaconda](https://www.anaconda.com/download)'s webpage and download the installer appropriate for your operating system.
2.  Open the downloaded installer.
3.  Click **Next** through the setup screens.
4.  Choose **Just Me** installation (recommended).
5.  Keep default settings and click **Install**.
6.  Click **Finish** when complete.

# Open anaconda prompt and run:
conda create --name DragSailEnv
conda activate DragSailEnv
conda install -c conda-forge orekit -y
conda install -c cyclus java-jdk
conda install conda-forge::openjdk
conda install plotly shapely numpy scipy pandas astropy matplotlib pyyaml geopandas -y

# Clone the repository
git clone git@github.com:dstannar/DragSail-Deorbit.git


