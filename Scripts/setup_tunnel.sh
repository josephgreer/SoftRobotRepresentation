ssh -o ServerAliveInterval=10 -i ./ami_keypair.pem -N -L 8889:localhost:8889 ubuntu@$AWS_DNS
