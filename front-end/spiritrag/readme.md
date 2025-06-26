# SpiritRAG Frontend

This is the frontend for the SpiritRAG project, which as an interactive chatbot designed for Q&A.

## Setup Instructions

### Prerequisites
- Ensure you have Node.js installed. You can download it from [Node.js Official Website](https://nodejs.org/).
- Install Vite globally if it's not already installed:
  ```bash
  npm install -g vite
  ```

### Steps to Start the Frontend

1. Navigate to the frontend directory:
   ```bash
   cd SpiritRAG/front-end/spiritrag
   ```

2. Install the required dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to the URL displayed in the terminal (usually `http://localhost:5173`).

## Notes
- The frontend communicates with the backend running on `http://127.0.0.1:5000`. Ensure the backend server is running before starting the frontend.
- If you encounter CORS issues, verify the backend's CORS configuration in `api-server.py`.

## Build for Production
To create a production build of the frontend, run:
```bash
npm run build
```
The production files will be generated in the `dist` folder.

## Troubleshooting
- **Port Conflicts**: If port `5173` is already in use, modify the `vite.config.js` file to use a different port.
- **Dependency Issues**: If you encounter issues during `npm install`, ensure your Node.js version is compatible with the dependencies listed in `package.json`.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.