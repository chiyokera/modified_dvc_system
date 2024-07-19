from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="/home/tanaka/path/to/SoccerNet")
mySNdl.downloadDataTask(task="caption-2024", split=["train","valid", "test","challenge"]) # SN challenge 2024ds