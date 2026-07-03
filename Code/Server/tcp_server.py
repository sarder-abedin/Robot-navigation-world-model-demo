import socket
import select
import threading
import fcntl
import struct
import queue

class TCPServer:
    def __init__(self):
        # Initialize server and client sockets
        self.server_socket = None
        self.client_sockets = {}
        # Guards client_sockets and active_connections against concurrent access
        # from the accept thread and the sender threads (AI broadcast + video).
        self._clients_lock = threading.RLock()
        # Message queue for incoming messages
        self.message_queue = queue.Queue()
        # Maximum number of clients allowed
        self.max_clients = 1
        # Current number of active connections
        self.active_connections = 0
        # Thread for accepting new connections
        self.accept_thread = None
        # Event to signal the server to stop
        self.stop_event = threading.Event()
        # Pipe for stopping the server
        self.stop_pipe_r, self.stop_pipe_w = socket.socketpair()
        self.stop_pipe_r.setblocking(0)
        self.stop_pipe_w.setblocking(0)

    def start(self, ip, port, max_clients=1, listen_count=1):
        # Set the maximum number of clients
        self.max_clients = max_clients
        # Create the server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((ip, port))
        self.server_socket.listen(listen_count)
        self.server_socket.setblocking(0)
        print(f"Server started, listening on {ip}:{port}")

        # Start the thread for accepting connections
        self.accept_thread = threading.Thread(target=self.accept_connections, daemon=True)
        self.accept_thread.start()

    def accept_connections(self):
        # Accept new connections until the server is stopped
        while not self.stop_event.is_set():
            # Snapshot the client sockets under the lock so a concurrent
            # remove_client()/close() from a sender thread cannot hand select()
            # a socket whose fileno() has become -1.
            with self._clients_lock:
                watch = [self.server_socket, self.stop_pipe_r] + list(self.client_sockets.keys())
            try:
                # Timeout so a client closed between the snapshot and the call
                # does not strand us; the next loop rebuilds a fresh watch list.
                readable, _, exceptional = select.select(watch, [], watch, 1.0)
            except (ValueError, OSError):
                # A watched fd was closed concurrently → drop any dead sockets
                # and retry with a fresh list.
                self._drop_dead_clients()
                continue
            for s in readable:
                if s == self.server_socket and self.active_connections < self.max_clients:
                    # Accept a new connection if the maximum number of clients is not reached
                    try:
                        client_socket, client_address = s.accept()
                    except OSError:
                        continue
                    client_socket.setblocking(0)
                    with self._clients_lock:
                        self.client_sockets[client_socket] = client_address
                        self.active_connections += 1
                    print(f"New connection from {client_address}, {self.active_connections} active connections.")
                elif s == self.server_socket and self.active_connections >= self.max_clients:
                    # Reject new connections if the maximum number of clients is reached
                    try:
                        client_socket, client_address = s.accept()
                        client_socket.close()
                        print(f"Rejected connection from {client_address}, max connections ({self.max_clients}) reached.")
                    except OSError:
                        pass
                elif s == self.stop_pipe_r:
                    # Stop the server if the stop pipe is read
                    self.stop_event.set()
                    break
                else:
                    try:
                        # Receive data from the client
                        data = s.recv(1024)
                        if data:
                            client_address = self.client_sockets.get(s)
                            if client_address is not None:
                                self.message_queue.put((client_address, data.decode('utf-8')))
                        else:
                            # Remove the client if no data is received
                            print(self.client_sockets.get(s), "disconnected")
                            self.remove_client(s)
                    except OSError as e:
                        if e.errno == 9 or e.errno == 32:
                            # Handle broken pipe / bad-fd errors
                            print(self.client_sockets.get(s), "disconnected")
                            self.remove_client(s)
                        else:
                            print(f"Unexpected error: {e}")
            for s in exceptional:
                # Handle exceptional conditions
                print(self.client_sockets.get(s), "disconnected")
                self.remove_client(s)
        print("Closing accept_connections...")

    def _drop_dead_clients(self):
        # Remove any client socket whose fileno() is invalid (closed elsewhere).
        with self._clients_lock:
            dead = [s for s in self.client_sockets if s.fileno() == -1]
        for s in dead:
            self.remove_client(s)

    def stop_pipe(self):
        # Send a byte to the stop pipe to signal the server to stop
        self.stop_pipe_w.send(b'\x00')

    def _send_all(self, client_socket, data, timeout=5.0):
        """
        Reliably send all bytes on a NON-BLOCKING socket.

        Client sockets are non-blocking (see accept_connections). A plain
        sendall() then raises BlockingIOError as soon as the kernel send buffer
        fills — which happens the moment the robot starts moving and the
        annotated JPEG frames get larger. The old code treated that transient
        condition as a fatal error and dropped the client, so the UI video went
        black and the connection "disconnected abruptly".

        Here we send in a loop and, when the buffer is full (BlockingIOError),
        wait for the socket to become writable again instead of giving up. We
        only raise (→ remove the client) on a genuinely broken or stuck peer.
        """
        view = memoryview(data)
        total = 0
        n = len(data)
        while total < n:
            try:
                sent = client_socket.send(view[total:])
                if sent == 0:
                    raise socket.error("connection broken during send")
                total += sent
            except BlockingIOError:
                # select can raise ValueError/OSError if the socket was closed
                # concurrently (accept thread) — let it propagate to the caller,
                # which treats it as a dead client.
                _, writable, _ = select.select([], [client_socket], [], timeout)
                if not writable:
                    raise socket.error("send timed out (client not draining)")

    def send_to_all_client(self, message):
        # Send a message to all connected clients
        if isinstance(message, str):
            message = message.encode('utf-8')
        with self._clients_lock:
            targets = list(self.client_sockets.keys())
        for client_socket in targets:
            try:
                self._send_all(client_socket, message)
            except (socket.error, OSError, ValueError) as e:
                print(f"Error sending data to {self.client_sockets.get(client_socket)}: {e}")
                self.remove_client(client_socket)

    def send_to_client(self, client_address, message):
        # Send a message to a specific client
        if isinstance(message, str):
            message = message.encode('utf-8')
        with self._clients_lock:
            targets = [(cs, addr) for cs, addr in self.client_sockets.items() if addr == client_address]
        if not targets:
            print(f"Client at {client_address} not found.")
            return
        for client_socket, addr in targets:
            try:
                self._send_all(client_socket, message)
            except (socket.error, OSError, ValueError) as e:
                print(f"Error sending data to {client_address}: {e}")
                self.remove_client(client_socket)

    def remove_client(self, client_socket):
        # Remove a client from the server. Idempotent and thread-safe: two
        # threads (accept + sender) can race to remove the same client on a
        # disconnect; pop-with-default avoids the KeyError that would kill the
        # accept thread, and the lock prevents active_connections corruption.
        with self._clients_lock:
            if self.client_sockets.pop(client_socket, None) is not None:
                self.active_connections -= 1
                closed = True
            else:
                closed = False
        if closed:
            try:
                client_socket.close()
            except Exception:
                pass

    def close(self):
        # Close the server and all client connections
        self.stop_pipe()
        if self.accept_thread is not None:
            self.accept_thread.join()
        if self.server_socket is not None:
            self.server_socket.close()
        for s in list(self.client_sockets):
            s.close()
        self.client_sockets.clear()
        print("Server stopped.")

    def get_client_ips(self):
        # Get a list of IP addresses of connected clients
        return [addr[0] for addr in self.client_sockets.values()]

def get_interface_ip():
    # Get the IP address of the specified network interface
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', b'wlan0'[:15]))[20:24])

if __name__ == "__main__":
    server = TCPServer()
    ip = get_interface_ip()
    port = 12345
    server.start(ip, port)

    try:
        while True:
            # Process incoming messages
            while not server.message_queue.empty():
                client_address, message = server.message_queue.get()
                print(f"Received message from {client_address}: {message}")
                server.send_to_client(client_address, message)
    except KeyboardInterrupt:
        print("Server interrupted by user.")
    finally:
        server.close()