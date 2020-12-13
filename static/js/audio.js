function play_audio() {
    var source = "../static/new.mp3"
    var audio = document.createElement("audio");
    //
    audio.autoplay = true;
    //
    audio.load()
    audio.addEventListener("load", function () {
        audio.play();
    }, true);
    audio.src = source;
}

play_audio()