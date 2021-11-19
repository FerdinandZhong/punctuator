# Notes

## TODO v0.2.0
[ ] add async inference
    [ ] set up batching inference server with mosec
        [ ] run mosec in subprocess
    [ ] provide two interface
        [ ] python interface
            [ ] within async inference func, await http request to mosec server and get result
            [ ] response to user
        [ ] http interface, user directly call mosec server

## TODO v1.0.0
[ ] build own batching backend to replace mosec to provide a main socket connection to python to receive request --> avoid http transportation of data