import threading
import time

# Server function
def server():
    print("Server started.")
    while True:
        # Wait for client updates or handle aggregation
        print("Server is waiting for client updates...")
        time.sleep(5)  # Simulate server running
        # Condition to break the loop and terminate gracefully (optional)

# Client function
def client(client_id):
    print(f"Client {client_id} started.")
    for i in range(3):  # Simulate local training rounds
        print(f"Client {client_id} is training round {i + 1}")
        time.sleep(2)  # Simulate training time
    print(f"Client {client_id} finished training.")

# Creating threads for server and clients
server_thread = threading.Thread(target=server, daemon=True)
client_threads = [threading.Thread(target=client, args=(i,)) for i in range(2)]

# Start server and client threads
server_thread.start()
for ct in client_threads:
    ct.start()

# Join client threads to wait for completion (server can run indefinitely)
for ct in client_threads:
    ct.join()

print("All clients finished. Server is still running.")
