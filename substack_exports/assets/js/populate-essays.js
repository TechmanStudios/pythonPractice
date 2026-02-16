let sortLikesAscending = false;
let sortDatesAscending = false;
let showHTML = true;

function sortEssaysByDate(data) {
    sortDatesAscending = !sortDatesAscending;
    return data.sort((a, b) =>
        sortDatesAscending
            ? new Date(a.date) - new Date(b.date)
            : new Date(b.date) - new Date(a.date)
    );
}

function sortEssaysByLikes(data) {
    sortLikesAscending = !sortLikesAscending;
    return data.sort((a, b) =>
        sortLikesAscending ? a.like_count - b.like_count : b.like_count - a.like_count
    );
}

function populateEssays(data) {
    const essaysContainer = document.getElementById('essays-container');
    const list = data
        .map(
            (essay) => `
        <li>
            <a href="../${showHTML ? essay.html_link : essay.file_link}" target="_blank">${essay.title}</a>
            <div class="subtitle">${essay.subtitle}</div>
            <div class="metadata">${essay.like_count} Likes - ${essay.date}</div>
        </li>
    `
        )
        .join('');
    essaysContainer.innerHTML = `<ul>${list}</ul>`;
}

function wireControls(essaysData) {
    const toggleButton = document.getElementById('toggle-format');
    if (toggleButton) {
        toggleButton.addEventListener('click', () => {
            showHTML = !showHTML;
            populateEssays(essaysData);
            toggleButton.textContent = showHTML ? 'Show Markdown' : 'Show HTML';
        });
    } else {
        showHTML = false;
    }

    document.getElementById('sort-by-date').addEventListener('click', () => {
        populateEssays(sortEssaysByDate([...essaysData]));
    });

    document.getElementById('sort-by-likes').addEventListener('click', () => {
        populateEssays(sortEssaysByLikes([...essaysData]));
    });
}

function bootstrap() {
    const embeddedDataElement = document.getElementById('essaysData');
    if (!embeddedDataElement) {
        return;
    }
    const essaysData = JSON.parse(embeddedDataElement.textContent);
    wireControls(essaysData);
    populateEssays(essaysData);
}

document.addEventListener('DOMContentLoaded', bootstrap);
