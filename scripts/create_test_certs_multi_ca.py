from datetime import UTC, datetime, timedelta
from tempfile import mkdtemp

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
#from cryptography.hazmat._oid import SERVER_AUTH, CLIENT_AUTH
from cryptography.hazmat._oid import ExtendedKeyUsageOID


from ipaddress import IPv4Address

temp_dir = 'certs' #mkdtemp()
print(temp_dir)



def genrsa(path):
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    with open(path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    return key


def create_cert(path, CN, C, ST, L, O, key):
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, C),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, ST),
            x509.NameAttribute(NameOID.LOCALITY_NAME, L),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, O),
            x509.NameAttribute(NameOID.COMMON_NAME, CN),
        ]
    )


    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC))
        .not_valid_after(
            # Our certificate will be valid for 10 days
            datetime.now(UTC)
            + timedelta(days=10)
        )
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    # Write our certificate out to disk.
    with open(path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert


def create_req(path, CN, CIP, C, ST, L, O, key):
    alt_names = [x509.IPAddress(IPv4Address(CIP))]
    
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(
            x509.Name(
                [
                    # Provide various details about who we are.
                    x509.NameAttribute(NameOID.COUNTRY_NAME, C),
                    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, ST),
                    x509.NameAttribute(NameOID.LOCALITY_NAME, L),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, O),
                    x509.NameAttribute(NameOID.COMMON_NAME, CN),
                ]
            )
        )
        .add_extension(x509.SubjectAlternativeName(alt_names),critical=True)
        .sign(key, hashes.SHA256())
    )
    with open(path, "wb") as f:
        f.write(csr.public_bytes(serialization.Encoding.PEM))
    return csr


def sign_certificate_request(path, CNIP, csr_cert, ca_cert, private_ca_key, server_auth=True, client_auth=False):
#    for ext in csr_cert.extensions:
#        print(f"EXT: {ext}")
#        if ext.oid._name == 'subjectAltName':
#            print(f"ADDING EXT: {ext}")
#            builder.add_extension(x509.SubjectAlternativeName([ext.value]),ext.critical)

    san = x509.SubjectAlternativeName([x509.IPAddress(IPv4Address(CNIP))])

    builder = (
       x509.CertificateBuilder()
       .subject_name(csr_cert.subject)
       .issuer_name(ca_cert.subject)
       .public_key(csr_cert.public_key())
       .serial_number(x509.random_serial_number())
       .add_extension(san,True)
       .not_valid_before(datetime.now(UTC))
       .not_valid_after(
            # Our certificate will be valid for 10 days
            datetime.now(UTC)
            + timedelta(days=10)
            # Sign our certificate with our private key
       )
    )

    if server_auth:
        builder.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), True)
    if client_auth:
        builder.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), True)

    cert = builder.sign(private_ca_key, hashes.SHA256())

    with open(path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert


def create_full_ca(ca_suffix):

    ca_key = genrsa(temp_dir + f"/ca_{ca_suffix}.key")
    ca_cert = create_cert(
        temp_dir + f"/ca_{ca_suffix}.crt",
        f"ca_{ca_suffix}",
        "US",
        "New York",
        "New York",
        "Gloo Certificate Authority",
        ca_key,
    )
    return (ca_cert,ca_key)

def cat_cas(cafilenames):
    with open(temp_dir + "/ca_all.pem", "w") as outfile:
        for fname in cafilenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


(ca_central_cert, ca_central_key) = create_full_ca("central")
(ca_hie1_cert, ca_hie1_key) = create_full_ca("hie1")
(ca_hie2_cert, ca_hie2_key) = create_full_ca("hie2")
(ca_hie3_cert, ca_hie3_key) = create_full_ca("hie3")

cat_cas([
		temp_dir + "/ca_central.crt",
		temp_dir + "/ca_hie1.crt",
		temp_dir + "/ca_hie2.crt",
		temp_dir + "/ca_hie3.crt",
#		temp_dir + "/hie1.pem",
#		temp_dir + "/hie2.pem",
#		temp_dir + "/hie3.pem",
	])

#		temp_dir + "/ca_central.key",
#		temp_dir + "/ca_hie1.key",
#		temp_dir + "/ca_hie2.key",
#		temp_dir + "/ca_hie3.key",

pkey = genrsa(temp_dir + "/central.key")
server_csr = create_req(
    temp_dir + "/central.csr",
    "central",
    "172.200.1.1",
    "US",
    "California",
    "San Francisco",
    "Gloo Testing Company",
    pkey,
)

pkey = genrsa(temp_dir + "/hie1.key")
hie1_csr = create_req(
    temp_dir + "/hie1.csr",
    "hie1",
    "172.200.1.2",
    "US",
    "California",
    "San Francisco",
    "Gloo Testing Company",
    pkey,
)

pkey = genrsa(temp_dir + "/hie2.key")
hie2_csr = create_req(
    temp_dir + "/hie2.csr",
    "hie2",
    "172.200.1.3",
    "US",
    "California",
    "San Francisco",
    "Gloo Testing Company",
    pkey,
)

pkey = genrsa(temp_dir + "/hie3.key")
hie3_csr = create_req(
    temp_dir + "/hie3.csr",
    "hie3",
    "172.200.1.4",
    "US",
    "California",
    "San Francisco",
    "Gloo Testing Company",
    pkey,
)

cert = sign_certificate_request(temp_dir + "/central.pem", "172.200.1.1", server_csr, ca_central_cert, ca_central_key, True, True)

cert = sign_certificate_request(temp_dir + "/hie1.pem", "172.200.1.2", hie1_csr, ca_hie1_cert, ca_hie1_key)
cert = sign_certificate_request(temp_dir + "/hie2.pem", "172.200.1.3", hie2_csr, ca_hie2_cert, ca_hie2_key)
cert = sign_certificate_request(temp_dir + "/hie3.pem", "172.200.1.4", hie3_csr, ca_hie3_cert, ca_hie3_key)

