#! usr/bin/python3
# -*- coding: utf-8 -*-
#
# Flicket - copyright Paul Bourne: evereux@gmail.com

from flask import (flash,
                   redirect,
                   url_for,
                   request,
                   session,
                   render_template,
                   g,
                   jsonify)
from flask_babel import gettext
from flask_login import login_required

from . import flicket_bp
from application import app
from application.flicket.forms.flicket_forms import CreateTicketForm
from application.flicket.models.flicket_models_ext import FlicketTicketExt

from application.utils.torch_utils import predict_sentiment

# create ticket


@flicket_bp.route(app.config['FLICKET'] + 'ticket_create/', methods=['GET', 'POST'])
@login_required
def ticket_create():
    # default category based on last submit (get from session)
    # using session, as information about last created ticket can be sensitive
    # in future it can be stored in extended user model instead
    last_category = session.get('ticket_create_last_category')
    form = CreateTicketForm(category=last_category)

    if form.validate_on_submit():
        new_ticket = FlicketTicketExt.create_ticket(title=form.title.data,
                                                    user=g.user,
                                                    content=form.content.data,
                                                    category=form.category.data,
                                                    priority=form.priority.data,
                                                    hours=form.hours.data,
                                                    files=request.files.getlist("file"))

        flash(gettext('New Ticket created.'), category='success')

        session['ticket_create_last_category'] = form.category.data

        return redirect(url_for('flicket_bp.ticket_view', ticket_id=new_ticket.id))

    title = gettext('Create Ticket')
    return render_template('flicket_create.html', title=title, form=form)


@flicket_bp.route(app.config['FLICKET'] + 'predict/', methods=['GET'])
@login_required
def predict():
    try:
        prediction = predict_sentiment(
            'Movie is the worst one I have ever seen!! The story has no meaning at all')
        data = {'prediction': prediction.item(
        ), 'class_name': str(prediction.item())}
        return jsonify(data)
    except:
        return jsonify({'error': 'error during prediction'})
    # return render_template('flicket_create.html', title=title, form=form)
