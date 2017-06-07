angular.module('MyBlog')
  .controller('MainPageCtrl', MainPageCtrl);

function Post(title, url, description, image_url, date) {
  this.title = title;
  this.url = url;
  this.short_description = description;
  this.image_url = image_url;
  this.date = date;
}

function MainPageCtrl() {

  this.posts = [];
  for (var i = 0; i < 4; i++) {
    this.posts[i] = new Post("Testing blog", "http://images.techhive.com/images/article/2016/06/machine_learning_ai-100665980-primary.idge.jpg", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer nunc erat, pulvinar ac consequat non, viverra ut leo.", "http://images.techhive.com/images/article/2016/06/machine_learning_ai-100665980-primary.idge.jpg", "Dec. 6, 2016")

  }
  console.log(this.posts)
}
