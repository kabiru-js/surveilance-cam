import socket
import struct
# 1. Create the socket object
# AF_INET means we are using a standard IPv4 address (like 127.0.0.1)
# SOCK_STREAM means we want TCP, which guarantees our messages arrive in order
radio = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
test_frame_size = 123456 
binary_header = struct.pack('!I',test_frame_size)
# 2. Prevent a common Windows error where the port stays "locked" if you restart your code
radio.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# 3. Lock this script onto our local address and channel 6000
radio.bind(('127.0.0.1', 5000))

print("Radio is alive")
# 4. Tell the radio to listen for an incoming call
radio.listen(1)
print("Waiting for someone to tune in ...")

# 5. This line is a 'blocking' line. The code will PAUSE right here.
# It will not move forward until a client connects.
# Once a client connects, it gives us two things:
#   connection = the active pipeline to talk through
#   address    = the network address of who connected
connection, address = radio.accept()
print(f"Success! Someone connected from: {address}")

# 6. Speak through the active connection pipeline.
# CRITICAL RULE: Network sockets ONLY understand raw bytes. 
# You cannot send regular text like "Hello". 
# You MUST put a lowercase 'b' in front of the quotes to convert it to bytes!
connection.sendall(binary_header)
print("Sent data safely across the wire.")

# 7. Hang up the phone
connection.close()
radio.close()
print("Connection closed. Script finished.")

