function mapStringToCodedElement(source,target) {
	target.setValue('code', source);
}

function mapGetAge(source,target) {
   var today = new Date();
    var birthDate = new Date(source.getValue());
    var age = today.getFullYear() - birthDate.getFullYear();
    var m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
        age--;
    }
   target.setValue( age);
}