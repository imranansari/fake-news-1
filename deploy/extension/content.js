console.log('loaded...');
let spanSelection = null;

function detectFakeNews() {
    return false;
}

document.addEventListener("mouseup", (event) => {
    if (spanSelection) {
        console.log(spanSelection);
        // Reset and remove span selection
        document.body.removeChild(spanSelection);
        spanSelection = null;
    }
    let text = ""
    if (window.getSelection) {
        text = window.getSelection().toString();
    } else if (document.selection && document.selection.type != "Control") {
        text = document.selection.createRange().text;
    }
    if (text === '') return
    const isFake = detectFakeNews(); 
    const imgURL = chrome.runtime.getURL("images/trump_amca_48.png");
    console.log(imgURL);
    console.log(event.clientX, event.clientY);
    const spanElem = document.createElement("span");
    spanElem.innerHTML = `
        <img class="img-sty" src=${imgURL} height=32 width=32> ${text}
    `;
    spanElem.className = "popup-tag";
    spanElem.style.display = "flex";
    spanElem.style.left = `${window.scrollX + event.clientX}px`;
    spanElem.style.top = `${window.scrollY + event.clientY}px`;
    if (isFake) {
        spanElem.style.backgroundColor = "red";
    } else {
        spanElem.style.backgroundColor = "#4be371";
    }
    document.body.appendChild(spanElem);
    spanSelection = spanElem;
    console.log("AFTER");
    
});
