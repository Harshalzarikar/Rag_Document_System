document.addEventListener('DOMContentLoaded', () => {
    const statusContainer = document.getElementById('status-container');
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const documentsList = document.getElementById('documents-list');
    const chatHistory = document.getElementById('chat-history');
    const chatInput = document.getElementById('chat-input');
    const chatButton = document.getElementById('chat-button');

    const API_URL = 'http://127.0.0.1:8001';

    async function fetchStatus() {
        try {
            const response = await fetch(`${API_URL}/`);
            const data = await response.json();
            statusContainer.innerHTML = `
                <div class="flex justify-between items-center">
                    <span class="font-medium text-gray-600">Status:</span>
                    <span class="font-bold ${data.status === 'healthy' ? 'text-green-600' : 'text-red-600'}">${data.status}</span>
                </div>
                <div class="flex justify-between items-center mt-2">
                    <span class="font-medium text-gray-600">Index Ready:</span>
                    <span class="font-bold ${data.index_ready ? 'text-green-600' : 'text-red-600'}">${data.index_ready}</span>
                </div>
                <div class="flex justify-between items-center mt-2">
                    <span class="font-medium text-gray-600">Documents:</span>
                    <span class="font-bold text-blue-600">${data.documents_count}</span>
                </div>
            `;
        } catch (error) {
            statusContainer.innerHTML = '<p class="text-red-500">Error fetching status</p>';
            console.error('Error fetching status:', error);
        }
    }

    async function fetchDocuments() {
        try {
            const response = await fetch(`${API_URL}/documents`);
            const data = await response.json();
            documentsList.innerHTML = '';
            if (data.documents.length === 0) {
                documentsList.innerHTML = '<p class="text-gray-500">No documents uploaded yet.</p>';
                return;
            }
            data.documents.forEach(doc => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span class="font-medium text-gray-800">${doc.filename}</span>
                    <span class="text-sm text-gray-500">(${(doc.size / 1024).toFixed(2)} KB)</span>
                `;
                documentsList.appendChild(li);
            });
        } catch (error) {
            console.error('Error fetching documents:', error);
        }
    }

    async function uploadFile() {
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file to upload.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            alert(data.message);
            fetchDocuments();
            fetchStatus();
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('File upload failed.');
        }
    }

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        appendMessage('user', message);
        chatInput.value = '';

        try {
            const response = await fetch(`${API_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });
            const data = await response.json();
            appendMessage('bot', data.response);
        } catch (error) {
            console.error('Error sending message:', error);
            appendMessage('bot', 'Sorry, something went wrong.');
        }
    }

    function appendMessage(sender, text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add(`${sender}-message`);
        messageElement.textContent = text;
        chatHistory.appendChild(messageElement);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    uploadButton.addEventListener('click', uploadFile);
    chatButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    fetchStatus();
    fetchDocuments();
});
