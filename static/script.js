async function submitImage() {
    const input = document.getElementById("imageInput");
    const file = input.files[0];

    const formData = new FormData();
    formData.append("image", file);

    const res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    document.getElementById("explanation").innerText = data.explanation;
}
