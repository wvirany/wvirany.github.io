// Function to update figure numbers
function updateFigureNumbers() {
    const figures = document.querySelectorAll('.figure');
    figures.forEach((figure, index) => {
        const caption = figure.querySelector('.figure-caption');
        if (caption) {
            caption.textContent = caption.textContent.replace(/Figure \d+:/, `Figure ${index + 1}:`);
        }
        figure.id = `fig${index + 1}`;
    });

    const figRefs = document.querySelectorAll('.fig-ref');
    figRefs.forEach(ref => {
        const figId = ref.getAttribute('href').slice(1);
        const figElement = document.getElementById(figId);
        if (figElement) {
            const figIndex = Array.from(figures).indexOf(figElement) + 1;
            ref.textContent = `Figure ${figIndex}`;
        }
    });
}

// Call the function when the page loads
window.addEventListener('load', updateFigureNumbers);