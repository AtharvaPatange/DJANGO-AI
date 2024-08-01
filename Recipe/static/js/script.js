// document.getElementById('apiForm').addEventListener('submit', async function (e) {
//     e.preventDefault();

//     let inputPrompt = document.getElementById('input_prompt').value;
//     let imageFile = document.getElementById('image').files[0];
//     let imageUrl = document.getElementById('image_url').value;

//     let imageStr = '';

//     if (imageFile) {
//         // Convert image file to base64
//         imageStr = await toBase64(imageFile);
//     } else if (imageUrl) {
//         imageStr = imageUrl;
//     }

//     let data = {
//         input_prompt: inputPrompt,
//         image_str: imageStr
//     };

//     let response = await fetch('/ats_home/', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify(data),
//     });

//     let result = await response.json();
//     document.getElementById('response').innerText = JSON.stringify(result, null, 2);
// });

// function toBase64(file) {
//     return new Promise((resolve, reject) => {
//         const reader = new FileReader();
//         reader.readAsDataURL(file);
//         reader.onload = () => resolve(reader.result.split(',')[1]);
//         reader.onerror = error => reject(error);
//     });
// }
document.getElementById('promptForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const inputPrompt = document.getElementById('inputPrompt').value;
    const imageURL = document.getElementById('imageURL').value;
    const imageFile = document.getElementById('imageFile').files[0];

    let imageBase64 = "";

    if (imageFile) {
        const reader = new FileReader();
        reader.onloadend = async function() {
            imageBase64 = reader.result.split(',')[1];
            await sendRequest(inputPrompt, imageURL, imageBase64);
        }
        reader.readAsDataURL(imageFile);
    } else {
        await sendRequest(inputPrompt, imageURL, imageBase64);
    }
});

async function sendRequest(inputPrompt, imageURL, imageBase64) {
    const responseContainer = document.getElementById('responseContainer');
    responseContainer.innerHTML = '';

    const payload = {
        input_prompt: inputPrompt,
        image_url: imageURL,
        image_base64: imageBase64
    };

    try {
        const response = await fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }

        const result = await response.json();
        responseContainer.innerHTML = `<p>${result.response}</p>`;
    } catch (error) {
        responseContainer.innerHTML = `<p>Error: ${error.message}</p>`;
    }
}
