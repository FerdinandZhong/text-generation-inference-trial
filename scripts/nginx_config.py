import argparse


def update_nginx_conf(conda_env):
    cfg = """
# nginx Configuration File
# https://www.nginx.com/resources/wiki/start/topics/examples/full/
# http://nginx.org/en/docs/dirindex.html
# https://www.nginx.com/resources/wiki/start/

# Run as a unique, less privileged user for security.
# user nginx www-data;  ## Default: nobody
user root;

# If using supervisord init system, do not run in deamon mode.
# Bear in mind that non-stop upgrade is not an option with "daemon off".
daemon off;

# Sets the worker threads to the number of CPU cores available in the system
# for best performance.
# Should be > the number of CPU cores.
# Maximum number of connections = worker_processes * worker_connections
worker_processes 4;  ## Default: 1

# Maximum number of open files per worker process.
# Should be > worker_connections.
# http://blog.martinfjordvald.com/2011/04/optimizing-nginx-for-high-traffic-loads/
# http://stackoverflow.com/a/8217856/2127762
# Each connection needs a filehandle (or 2 if you are proxying).
worker_rlimit_nofile 8192;

events {
  # If you need more connections than this, you start optimizing your OS.
  # That's probably the point at which you hire people who are smarter than
  # you as this is *a lot* of requests.
  # Should be < worker_rlimit_nofile.
  worker_connections 8000;
}

# Log errors and warnings to this file
# This is only used when you don't override it on a server{} level
#error_log  logs/error.log  notice;
#error_log  logs/error.log  info;
error_log  var/log/nginx/error.log warn;

# The file storing the process ID of the main process
pid var/run/nginx.pid;


http {
  # Log access to this file
  # This is only used when you don't override it on a server{} level
  access_log var/log/nginx/access.log;

  # Hide nginx version information.
  server_tokens off;

  # Controls the maximum length of a virtual host entry (ie the length
  # of the domain name).
  server_names_hash_bucket_size 64;

  # Specify MIME types for files.
  include mime.types;
  default_type application/octet-stream;

  # How long to allow each connection to stay idle.
  # Longer values are better for each individual client, particularly for SSL,
  # but means that worker connections are tied up longer.
  keepalive_timeout 20s;

  # Speed up file transfers by using sendfile() to copy directly
  # between descriptors rather than using read()/write().
  # For performance reasons, on FreeBSD systems w/ ZFS
  # this option should be disabled as ZFS's ARC caches
  # frequently used files in RAM by default.
  sendfile on;

  # Don't send out partial frames; this increases throughput
  # since TCP frames are filled up before being sent out.
  tcp_nopush on;

  # Enable gzip compression.
  gzip on;

  # Compression level (1-9).
  # 5 is a perfect compromise between size and CPU usage, offering about
  # 75% reduction for most ASCII files (almost identical to level 9).
  gzip_comp_level 5;

  # Don't compress anything that's already small and unlikely to shrink much
  # if at all (the default is 20 bytes, which is bad as that usually leads to
  # larger files after gzipping).
  gzip_min_length 256;

  # Compress data even for clients that are connecting to us via proxies,
  # identified by the "Via" header (required for CloudFront).
  gzip_proxied any;

  # Tell proxies to cache both the gzipped and regular version of a resource
  # whenever the client's Accept-Encoding capabilities header varies;
  # Avoids the issue where a non-gzip capable client (which is extremely rare
  # today) would display gibberish if their proxy gave them the gzipped version.
  gzip_vary on;

  # Compress all output labeled with one of the following MIME-types.
  gzip_types
    application/atom+xml
    application/javascript
    application/json
    application/ld+json
    application/manifest+json
    application/rss+xml
    application/vnd.geo+json
    application/vnd.ms-fontobject
    application/x-font-ttf
    application/x-web-app-manifest+json
    application/xhtml+xml
    application/xml
    font/opentype
    image/bmp
    image/svg+xml
    image/x-icon
    text/cache-manifest
    text/css
    text/plain
    text/vcard
    text/vnd.rim.location.xloc
    text/vtt
    text/x-component
    text/x-cross-domain-policy;
  # text/html is always compressed by gzip module

  # This should be turned on if you are going to have pre-compressed copies (.gz) of
  # static files available. If not it should be left off as it will cause extra I/O
  # for the check. It is best if you enable this in a location{} block for
  # a specific directory, or on an individual server{} level.
  # gzip_static on;

  include sites.d/*.conf;

} 
    """
    nginx_config_path = f"/root/miniconda3/envs/{conda_env}/etc/nginx/nginx.conf"
    with open(nginx_config_path, "w") as fh:
        fh.writelines(cfg)


def update_defaultsite_conf(conda_env):
    cfg = """
upstream dashboard_servers {
    server 127.0.0.1:8265;
}    

upstream app_servers {
    server 127.0.0.1:5000;
}

server {
    listen 127.0.0.1:16002;

    location / {
        add_header Content-Type text/plain;
        return 200 'Okay!';
    }

    location /dashboard/ {
        proxy_pass http://dashboard_servers/;
    }

    location /generate {
        proxy_pass http://app_servers/;
    }

    # redirect server error pages to the static page /50x.html
    error_page   500 502 503 504  /50x.html;
    location = /50x.html {
        root   etc/nginx/default-site/;
    }
}
    """
    nginx_config_path = (
        f"/root/miniconda3/envs/{conda_env}/etc/nginx/sites.d/default-site.conf"
    )

    with open(nginx_config_path, "w") as fh:
        fh.writelines(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--conda_env", default="server-env")

    conda_env = ap.parse_args().conda_env

    update_defaultsite_conf(conda_env=conda_env)
    update_nginx_conf(conda_env=conda_env)
