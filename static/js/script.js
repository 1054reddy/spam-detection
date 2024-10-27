document.getElementById('emailForm').addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent form submission

    const emailText = document.getElementById('emailText').value;

    fetch('/check_spam', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'email_text': emailText
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = data.result;
    })
    .catch(error => console.error('Error:', error));
});
