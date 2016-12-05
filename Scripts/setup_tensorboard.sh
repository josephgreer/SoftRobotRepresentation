ssh -o ServerAliveInterval=10 -i ./ami_keypair.pem -N -L 6006:localhost:6006 ubuntu@$AWS_DNS
