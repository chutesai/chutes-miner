# Nginx image that runs as non-root (UID 1000) for use with runAsNonRoot pod security.
# Config is mounted from ConfigMap; pid and temp paths use /var/cache/nginx (writable by 1000).
FROM nginx:1.27 AS base

RUN set -e; \
  groupadd -g 1000 chutes; \
  useradd -u 1000 -g chutes -m -s /bin/sh chutes; \
  mkdir -p /var/cache/nginx/client_temp /var/cache/nginx/proxy_temp \
    /var/cache/nginx/fastcgi_temp /var/cache/nginx/uwsgi_temp /var/cache/nginx/scgi_temp \
    /var/log/nginx; \
  chown -R chutes:chutes /var/cache/nginx /var/log/nginx

# Bypass default entrypoint (it writes to /etc/nginx/conf.d as root). Config is mounted.
USER chutes
EXPOSE 80
ENTRYPOINT ["nginx", "-g", "daemon off;"]

FROM base AS production

FROM base AS development
