Make sure you have docker installed and docker daemon running. 

To execute it: Go to Code Of Winning Methods folder and run this.

`sudo docker compose -f infra/compose.yaml up --build`

After it, if you want to enter the container via bash or execute the python scripts:
 `sudo docker compose -f infra/compose.yaml exec app /bin/bash`

The container automatically runs in `localhost:8888` so you might want to execute the notebooks there.

To stop the container
`sudo docker compose -f infra/compose.yaml down`

To restart the container:
`sudo docker compose -f infra/compose.yaml up -d`