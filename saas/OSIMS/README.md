# sturdy-robot
smartkit.ai

To start, use `docker-compose up --build -d`.
To stop, use `docker-compose down`.

As described in the `docker-compose.yml` file, Orthanc's HTTP server is
reachable via port 80 on the Docker host, and Orthanc's DICOM server is
reachable via port 104 on the Docker host.

### Reference

http://www.osimis.io/en/blog/2016/06/03/medical-imaging-straight-from-your-browser.html
