'use static';
// 
function _(ele){
    return document.getElementById(ele);
}


let fileData = new FormData();
function showPreview(event) {
  if (event.target.files.length > 0) {
    console.log(event);
    console.log("File check");
    var Classification = document.getElementById("tclassi");
    fileData.append("image", event.target.files[0]);
    let src = URL.createObjectURL(event.target.files[0]);
    let preview = document.getElementById("imageshow");
    // Classification.classList.remove("disabled");
    console.log(preview);
    preview.src = src;
    // preview.style.display = "block";
    preview.classList.remove("d-none");
    
  }
};


if(_("inputPath").textContent!=='')
{
    _("imageshow").classList.remove("d-none");
};
if(_("limePath").textContent!=='')
{
    _("photo").classList.remove("d-none");
};