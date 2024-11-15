### Docker Compose Configuration

#### Networks

| Key                                                        | Value          |
| ---------------------------------------------------------- | -------------- |
| **public-net**                                             |                |
| &nbsp;&nbsp;**driver**                                     | bridge         |
| &nbsp;&nbsp;**enable_ipv6**                                | false          |
| &nbsp;&nbsp;**ipam**                                       |                |
| &nbsp;&nbsp;&nbsp;&nbsp;**driver**                         | default        |
| &nbsp;&nbsp;&nbsp;&nbsp;**config**                         |                |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**subnet**             | 172.200.0.0/16 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**ip_range**           | 172.200.1.0/24 |
| &nbsp;&nbsp;**driver_opts**                                |                |
| &nbsp;&nbsp;&nbsp;&nbsp;**com.docker.network.bridge.name** | central_public |

#### Services

| Key                                                                            | Default              | Supported                                                      |
| ------------------------------------------------------------------------------ | -------------------- | -------------------------------------------------------------- |
| **central**                                                                    |                      |                                                                |
| &nbsp;&nbsp;**build**                                                          |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**context**                                            | .                    |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**target**                                             | central              |                                                                |
| &nbsp;&nbsp;**privileged**                                                     | true                 |                                                                |
| &nbsp;&nbsp;**hostname**                                                       | central              |                                                                |
| &nbsp;&nbsp;**container_name**                                                 | central              |                                                                |
| &nbsp;&nbsp;**restart**                                                        | on-failure           |                                                                |
| &nbsp;&nbsp;**environment**                                                    |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**WORLD_SIZE**                                         | 3                    | >= 1                                                           |
| &nbsp;&nbsp;&nbsp;&nbsp;**CENTRAL_ADDRESS**                                    | 172.200.1.1          |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**CENTRAL_PORT**                                       | 7732                 |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**IFNAME**                                             | eth0                 |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**CONFIG**                                             | os                   |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**TZ**                                                 | America/Indianapolis | lookup https://mljar.com/blog/list-pytz-timezones/ for options |
| &nbsp;&nbsp;&nbsp;&nbsp;**MODEL_PATH**                                         | saves                | s3://bucket/key                                                |
| &nbsp;&nbsp;**volumes**                                                        |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**./tcpdump-logs/central:/tcpdump-logs**               |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**./stunnel-logs/central:/var/log/stunnel**            |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**./certs/ca_all.pem:/etc/stunnel/certs/ca.pem**       |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**./certs/central.pem:/etc/stunnel/certs/central.pem** |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**./certs/central.key:/etc/stunnel/certs/central.key** |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**./saves/:/saves**                                    |                      |                                                                |
| &nbsp;&nbsp;**ports**                                                          |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**5000:5000**                                          | 5000:5000            |                                                                |
| &nbsp;&nbsp;**networks**                                                       |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**public-net**                                         | public-net           |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**ipv4_address**                                       | 172.200.1.1          |                                                                |
| &nbsp;&nbsp;**deploy**                                                         |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;**resources**                                          |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**reservations**                           |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**devices**                    |                      |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**driver**         | nvidia               |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**count**          | 1                    |                                                                |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**capabilities**   | [ gpu ]              |                                                                |

