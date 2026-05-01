const fs = require('fs');
const path = require('path');

const publicDir = path.join(__dirname, 'public');

console.log('Build started...');
console.log('Public dir:', publicDir);

const files = [
    'index.html',
    'css/style.css',
    'js/api.js',
    'js/main.js'
];

files.forEach(file => {
    const src = path.join(publicDir, file);
    const dest = path.join(publicDir, file);
    if (fs.existsSync(src)) {
        console.log(`OK: ${file} (${fs.statSync(src).size} bytes)`);
    } else {
        console.log(`MISSING: ${file}`);
    }
});

console.log('Build complete!');
